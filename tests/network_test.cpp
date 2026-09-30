#include "kan/network.hpp"
#include "support/test.hpp"
#include <limits>
#include <numeric>

namespace {
kan::Network fixture() {
    kan::BasisConfig c;c.size=3;kan::Layer first(2,3,c);
    c.kind=kan::BasisKind::Legendre;kan::Layer second(3,1,c);
    for(auto* layer:{&first,&second}) {
        std::vector<double> p(layer->coefficients().size()),b(layer->bias().size(),0.05);
        for(std::size_t i=0;i<p.size();++i)p[i]=0.04*(static_cast<double>(i%5)-2);
        layer->set_parameters(p,b);
    }
    return kan::Network({first,second});
}
double objective(const kan::Network& net,const std::vector<double>& x,const std::vector<double>& g) {
    const auto y=net.forward(x,2);return std::inner_product(y.begin(),y.end(),g.begin(),0.0);
}
}
TEST(topology_must_be_nonempty_and_compatible) {
    test::throws<std::invalid_argument>([]{kan::Network n({});});
    test::throws<std::invalid_argument>([]{kan::Network n({kan::Layer(2,3,{}),kan::Layer(2,1,{})});});
}
TEST(forward_is_layer_composition) {
    auto net=fixture();const std::vector<double>x{-0.4,0.2,0.7,-0.1};
    const auto first=net.layers()[0].forward(x,2),expected=net.layers()[1].forward(first,2),actual=net.forward(x,2);
    REQUIRE(actual==expected);
}
TEST(network_gradients_match_finite_differences) {
    auto net=fixture();std::vector<double>x{-0.4,0.2,0.7,-0.1},g{0.8,-0.3};const auto grad=net.backward(x,2,g);double h=1e-6;
    REQUIRE(grad.layers.size()==2);
    for(std::size_t i=0;i<x.size();++i){auto p=x,m=x;p[i]+=h;m[i]-=h;test::near(grad.input[i],(objective(net,p,g)-objective(net,m,g))/(2*h),1e-7);}
    for(std::size_t l=0;l<net.layers().size();++l){
        std::vector<kan::Layer> layers(net.layers().begin(),net.layers().end());
        const std::vector<double> c(layers[l].coefficients().begin(),layers[l].coefficients().end()),b(layers[l].bias().begin(),layers[l].bias().end());
        for(std::size_t i=0;i<c.size();++i){auto plus=layers,minus=layers;auto pc=c,mc=c;pc[i]+=h;mc[i]-=h;plus[l].set_parameters(pc,b);minus[l].set_parameters(mc,b);
            test::near(grad.layers[l].coefficients[i],(objective(kan::Network(plus),x,g)-objective(kan::Network(minus),x,g))/(2*h),1e-7);}
        for(std::size_t i=0;i<b.size();++i){auto plus=layers,minus=layers;auto pb=b,mb=b;pb[i]+=h;mb[i]-=h;plus[l].set_parameters(c,pb);minus[l].set_parameters(c,mb);
            test::near(grad.layers[l].bias[i],(objective(kan::Network(plus),x,g)-objective(kan::Network(minus),x,g))/(2*h),1e-7);}
    }
}
TEST(empty_and_invalid_network_batches) {
    auto net=fixture();REQUIRE(net.forward({},0).empty());const auto grad=net.backward({},0,{});REQUIRE(grad.input.empty());REQUIRE(grad.layers.size()==2);
    for(const auto& layer:grad.layers)for(double v:layer.coefficients)test::near(v,0);
    test::throws<std::invalid_argument>([&]{net.forward(std::vector<double>{1},1);});
    test::throws<std::invalid_argument>([&]{net.backward(std::vector<double>{1,2},1,std::vector<double>{1,2});});
}
TEST(network_sgd_is_atomic_across_layers) {
    auto net=fixture();const std::vector<double>x{0.2,0.4};const auto before=net.forward(x,1);auto grad=net.backward(x,1,std::vector<double>{1});
    auto bad=grad;bad.layers.pop_back();test::throws<std::invalid_argument>([&]{net.sgd(bad,0.1);});
    bad=grad;bad.layers.back().bias[0]=std::numeric_limits<double>::quiet_NaN();test::throws<std::invalid_argument>([&]{net.sgd(bad,0.1);});
    REQUIRE(net.forward(x,1)==before);
    bad=grad;bad.layers.back().coefficients.back()=std::numeric_limits<double>::max();test::throws<std::overflow_error>([&]{net.sgd(bad,2);});
    REQUIRE(net.forward(x,1)==before);net.sgd(grad,0.1);REQUIRE(net.forward(x,1)!=before);
}
TEST(single_layer_network_matches_layer) {
    kan::Layer layer(1,1,{});kan::Network net({layer});const std::vector<double>x{0.1,0.2},g{1,2};
    REQUIRE(net.forward(x,2)==layer.forward(x,2));REQUIRE(net.backward(x,2,g).layers[0].coefficients==layer.backward(x,2,g).coefficients);
}
TEST(sgd_training_fits_polynomial) {
    kan::BasisConfig config;config.size=3;kan::Network net({kan::Layer(1,1,config)});std::vector<double>x(33),target(33);
    for(std::size_t i=0;i<x.size();++i){x[i]=-1+2*static_cast<double>(i)/32;target[i]=0.2+0.7*x[i]-0.4*x[i]*x[i];}
    for(int epoch=0;epoch<400;++epoch){auto g=net.forward(x,x.size());for(std::size_t i=0;i<g.size();++i)g[i]=2*(g[i]-target[i])/static_cast<double>(g.size());net.sgd(net.backward(x,x.size(),g),0.1);}
    const auto y=net.forward(x,x.size());double mse=0;for(std::size_t i=0;i<y.size();++i)mse+=(y[i]-target[i])*(y[i]-target[i]);REQUIRE(mse/y.size()<1e-12);
}
TEST(moved_from_network_and_layers_are_rejected) {
    auto source=fixture();auto destination=std::move(source);
    REQUIRE(destination.forward(std::vector<double>{0.1,0.2},1).size()==1);
    test::throws<std::invalid_argument>([&]{source.forward(std::vector<double>{0.1,0.2},1);});
    test::throws<std::invalid_argument>([&]{source.backward(std::vector<double>{0.1,0.2},1,std::vector<double>{1});});
    test::throws<std::invalid_argument>([&]{source.sgd({},0.1);});
    source=fixture();REQUIRE(source.forward(std::vector<double>{0.1,0.2},1).size()==1);
    kan::Layer layer(2,1,{});auto moved=std::move(layer);
    test::throws<std::invalid_argument>([&]{kan::Network invalid({layer});});
    REQUIRE(moved.inputs()==2);
}
int main(){return test::run();}
