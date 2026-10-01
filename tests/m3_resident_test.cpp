#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#include "support/test.hpp"
#include <limits>

namespace {
void compare(std::span<const double> a, std::span<const double> b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) test::near(a[i], b[i], 4e-10);
}
kan::BasisConfig config(kan::BasisKind kind) {
    kan::BasisConfig b{kind, 5}; b.centers = {-1,-0.5,0,0.5,1};
    b.scales = {0.3,0.5,0.8,1.1,1.5}; b.degree = 2;
    b.knots = {-1,-1,-1,-0.25,0.5,1,1,1};
    b.trainable_rbf = kind == kan::BasisKind::GaussianRbf;
    b.log_widths = {-0.8,-0.4,0,0.2,0.4}; return b;
}
kan::Network model(kan::BasisKind kind) {
    kan::Layer first(2,3,config(kind)), second(3,1,{kan::BasisKind::Chebyshev,3});
    for (auto* l : {&first,&second}) {
        std::vector<double> c(l->coefficients().size()), b(l->outputs(),0.01);
        for (std::size_t i=0;i<c.size();++i) c[i]=(static_cast<double>(i%11)-5)/90;
        l->set_parameters(c,b);
    }
    return kan::Network({first,second});
}
void parity(kan::BasisKind kind) {
    auto cpu=model(kind); kan::cuda::ResidentNetwork gpu(cpu,8);
    const auto allocations=gpu.workspace_allocations();
    const std::vector<double> x{-1,1,-0.25,0.5,1.2,-1.2,0,0.2}, dy{0.1,-0.3,0.2,0.4};
    gpu.upload_input(x,4); gpu.upload_output_gradient(dy);
    for(int step=0;step<3;++step) {
        gpu.forward(); compare(gpu.download_output(),cpu.forward(x,4));
        gpu.backward(); auto a=gpu.download_gradients(); auto e=cpu.backward(x,4,dy);
        compare(a.input,e.input);
        for(std::size_t j=0;j<a.layers.size();++j) {
            compare(a.layers[j].coefficients,e.layers[j].coefficients);
            compare(a.layers[j].bias,e.layers[j].bias);
            compare(a.layers[j].centers,e.layers[j].centers);
            compare(a.layers[j].log_widths,e.layers[j].log_widths);
        }
        gpu.sgd(0.03); cpu.sgd(e,0.03);
    }
    const auto snapshot=gpu.download_parameters(); compare(snapshot.forward(x,4),cpu.forward(x,4));
    if(kind==kan::BasisKind::GaussianRbf) {
        compare(snapshot.layers()[0].basis().centers,cpu.layers()[0].basis().centers);
        compare(snapshot.layers()[0].basis().log_widths,cpu.layers()[0].basis().log_widths);
    }
    REQUIRE(allocations==gpu.workspace_allocations());
}
}
TEST(m3_resident_spline_independent_hat_values) {
    kan::BasisConfig b{kan::BasisKind::BSpline,3}; b.degree=1;b.knots={-1,-1,0,1,1};
    kan::Layer l(1,3,b);l.set_parameters(std::vector<double>{1,0,0,0,1,0,0,0,1},std::vector<double>{0,0,0});
    kan::cuda::ResidentNetwork gpu(kan::Network({l}),3);
    gpu.upload_input(std::vector<double>{-0.5,0,1},3);gpu.forward();
    compare(gpu.download_output(),std::vector<double>{0.5,0.5,0,0,1,0,0,0,1});
}
TEST(m3_resident_mixed_families_and_learned_parameter_snapshots) {
    for(auto kind:{kan::BasisKind::BSpline,kan::BasisKind::MexicanHat,kan::BasisKind::GaussianRbf}) parity(kind);
}
TEST(m3_resident_zero_batch_regularization_and_validation) {
    auto cpu=model(kan::BasisKind::GaussianRbf); kan::cuda::ResidentNetwork gpu(cpu,0);
    gpu.upload_input({},0); gpu.upload_output_gradient({}); gpu.forward();
    test::throws<std::invalid_argument>([&]{gpu.backward(-1);});
    test::throws<std::invalid_argument>([&]{gpu.backward(std::numeric_limits<double>::infinity());});
    gpu.backward(0.3); const auto g=gpu.download_gradients(), e=cpu.regularization(0.3).gradients;
    REQUIRE(g.input.empty());
    for(std::size_t j=0;j<g.layers.size();++j) {
        compare(g.layers[j].coefficients,e.layers[j].coefficients);
        compare(g.layers[j].bias,e.layers[j].bias);
        compare(g.layers[j].centers,e.layers[j].centers); compare(g.layers[j].log_widths,e.layers[j].log_widths);
    }
    gpu.sgd(0.1); cpu.sgd(e,0.1); compare(gpu.download_parameters().layers()[0].coefficients(),cpu.layers()[0].coefficients());
}
TEST(m3_resident_width_candidate_validation_is_atomic) {
    kan::BasisConfig b{kan::BasisKind::GaussianRbf,1}; b.centers={0}; b.trainable_rbf=true; b.log_widths={0};
    kan::Layer first(1,1,b); first.set_parameters(std::vector<double>{1},std::vector<double>{0});
    kan::Layer second(1,1,{kan::BasisKind::Chebyshev,2}); second.set_parameters(std::vector<double>{0,1},std::vector<double>{0});
    kan::cuda::ResidentNetwork gpu(kan::Network({first,second}),1);
    gpu.upload_input(std::vector<double>{1},1); gpu.upload_output_gradient(std::vector<double>{1}); gpu.forward();gpu.backward();
    test::throws<std::overflow_error>([&]{gpu.sgd(2000);});
    const auto same=gpu.download_parameters(); compare(same.layers()[0].coefficients(),first.coefficients());
    compare(same.layers()[1].coefficients(),second.coefficients()); compare(same.layers()[0].basis().log_widths,b.log_widths);
    gpu.sgd(0.01); REQUIRE(gpu.download_parameters().layers()[0].basis().log_widths[0]<0);
}
TEST(m3_resident_explicit_refinement_reconstruction) {
    auto cpu=model(kan::BasisKind::BSpline); kan::cuda::ResidentNetwork before(cpu,2);
    const std::vector<double> x{-0.7,0.2,0.9,-0.1}; before.upload_input(x,2);before.forward();const auto expected=before.download_output();
    auto snapshot=before.download_parameters(); snapshot.insert_knot(0,0.1);
    kan::cuda::ResidentNetwork after(snapshot,2);after.upload_input(x,2);after.forward();compare(after.download_output(),expected);
}
int main() {if(!kan::cuda::available()) return 1;return test::run();}
