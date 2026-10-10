// Complete resident/host-batch MSE train steps, correctness and long learning
// runs. All mode/tile changes precede construction; each process owns its GPU.
#include "kan/resident.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

struct Case { const char* name; std::vector<std::size_t> widths; std::size_t batch; };
const std::vector<Case> cases = {
    {"tiny",{16,24,8},1024}, {"small",{64,64,32,16},1024},
    {"irregular",{63,95,17},257}, {"deep",{64,64,64,64,64,64,16},1024},
    {"medium",{256,256,256,10},2048}, {"wide",{256,256,256,10},8192},
    {"large",{1024,1024,1024},4096},
    // Large contractions with tails exercise virtual operands, shared
    // epilogues and partial final K tiles, independently of the timed cases.
    {"tail",{193,257,31},513}
};
void set_mode(int mode,int tile) {
#ifdef _WIN32
    _putenv_s("KAN_EXPERIMENT_MODE",std::to_string(mode).c_str());
    _putenv_s("KAN_EXPERIMENT_TILE",std::to_string(tile).c_str());
#else
    setenv("KAN_EXPERIMENT_MODE",std::to_string(mode).c_str(),1);
    setenv("KAN_EXPERIMENT_TILE",std::to_string(tile).c_str(),1);
#endif
}
std::vector<double> wave(std::size_t n,double scale,double freq,double phase) {
    std::vector<double> v(n);
    for(std::size_t i=0;i<n;++i) v[i]=scale*std::sin(freq*double(i)+phase);
    return v;
}
kan::Network network(const Case& c,bool branch,int seed) {
    std::vector<kan::NetworkLayer> layers;
    std::uint64_t state=12345+seed*173;
    auto uniform=[&]{state=state*6364136223846793005ULL+1442695040888963407ULL;return ((state>>11)+.5)/9007199254740992.0;};
    auto normal=[&]{return std::sqrt(-2*std::log(uniform()))*std::cos(6.283185307179586*uniform());};
    for(std::size_t j=0;j+1<c.widths.size();++j) {
        const auto in=c.widths[j],out=c.widths[j+1];
        kan::Layer layer(in,out,kan::ChebyshevConfig{7});
        std::vector<double> coeff(in*out*7),bias(out);
        for(auto& v:coeff) v=normal()*.1/std::sqrt(double(in*7));
        for(auto& v:bias) v=normal()*.01;
        layer.set_parameters(coeff,bias);
        if(branch) {
            auto weights=wave(in*out,.1/std::sqrt(double(in)),.917,.3+seed*.1);
            layer.set_residual(kan::SiluResidual{std::move(weights)});
        }
        layers.emplace_back(std::move(layer));
    }
    return kan::Network(std::move(layers));
}
std::vector<double> parameters(const kan::Network& net) {
    std::vector<double> result;
    for(const auto& stage:net.layers()) {
        const auto& l=std::get<kan::Layer>(stage);
        result.insert(result.end(),l.coefficients().begin(),l.coefficients().end());
        result.insert(result.end(),l.bias().begin(),l.bias().end());
        if(l.residual()) result.insert(result.end(),l.residual()->weights.begin(),l.residual()->weights.end());
    }
    return result;
}
std::vector<double> gradients(const kan::NetworkGradients& g) {
    auto result=g.input;
    for(const auto& stage:g.layers) {
        const auto& l=std::get<kan::LayerGradients>(stage);
        result.insert(result.end(),l.coefficients.begin(),l.coefficients.end());
        result.insert(result.end(),l.bias.begin(),l.bias.end());
        result.insert(result.end(),l.residual.begin(),l.residual.end());
    }
    return result;
}
double compare(const std::vector<double>& a,const std::vector<double>& b,const char* what,double tolerance) {
    if(a.size()!=b.size()) throw std::runtime_error("shape mismatch");
    double abs=0,norm=0;std::size_t changed=0;
    for(std::size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("nonfinite comparison");
        const auto delta=std::abs(a[i]-b[i]);
        abs=std::max(abs,delta);norm=std::max(norm,delta/(1+std::abs(a[i])));changed+=a[i]!=b[i];
    }
    std::printf("check,%s,%zu,%.9g,%.9g,%zu,%s\n",what,a.size(),abs,norm,changed,norm<=tolerance?"PASS":"FAIL");
    if(norm>tolerance) throw std::runtime_error(std::string(what)+" outside experiment tolerance");
    return norm;
}
int main(int argc,char** argv) try {
    if(argc<8) throw std::runtime_error("usage: cutlass_experiment check|bench|train mode tile f32|tf32|f64 resident|host case branch [seed windows steps]");
    const std::string action=argv[1],precision_name=argv[4],protocol=argv[5],case_name=argv[6];
    const int mode=std::stoi(argv[2]),tile=std::stoi(argv[3]),seed=argc>8?std::stoi(argv[8]):0;
    const bool branch=std::stoi(argv[7])!=0;
    const int windows=argc>9?std::stoi(argv[9]):5,steps=argc>10?std::stoi(argv[10]):200;
    auto it=std::find_if(cases.begin(),cases.end(),[&](const auto& c){return c.name==case_name;});
    if(it==cases.end()||steps<1||windows<1) throw std::runtime_error("invalid case or sample count");
    const auto& c=*it;
    const auto precision=precision_name=="f32"?kan::cuda::Precision::Float32:
        precision_name=="tf32"?kan::cuda::Precision::TensorFloat32:kan::cuda::Precision::Float64;
    auto net=network(c,branch,seed);
    std::vector<std::vector<double>> xs,ts;
    for(int b=0;b<4;++b) {
        xs.push_back(wave(c.batch*c.widths.front(),.95,.113+.017*b,.3*b+seed*.07));
        ts.push_back(wave(c.batch*c.widths.back(),.1,.071+.013*b,.7*b+seed*.11));
    }
    const auto up=wave(c.batch*c.widths.back(),1.0/c.batch,.07,.1);
    if(action=="check") {
        const double tolerance=precision_name=="tf32"?5e-3:precision_name=="f64"?0:1e-4;
        set_mode(0,tile);kan::cuda::ResidentNetwork ref(net,c.batch,precision);
        set_mode(mode,tile);kan::cuda::ResidentNetwork candidate(net,c.batch,precision);
        auto prepare=[&](auto& r){r.upload_input(xs[0],c.batch);r.upload_output_gradient(up);r.forward();};
        prepare(ref);prepare(candidate);
        compare(ref.download_output(),candidate.download_output(),"output",tolerance);
        ref.backward(1e-4);candidate.backward(1e-4);
        compare(gradients(ref.download_gradients()),gradients(candidate.download_gradients()),"gradients",tolerance);
        ref.sgd(.01);candidate.sgd(.01);
        compare(parameters(ref.download_parameters()),parameters(candidate.download_parameters()),"SGD",tolerance);
        candidate.upload_parameters(net);
        candidate.upload_input(xs[0],c.batch);candidate.upload_output_gradient(up);
        candidate.train_step(.01,1e-4);
        auto graph=parameters(candidate.download_parameters());
        set_mode(mode,tile);kan::cuda::ResidentNetwork eager(net,c.batch,precision);
        prepare(eager);eager.backward(1e-4);eager.sgd(.01);
        compare(parameters(eager.download_parameters()),graph,"graph-eager",0);
        candidate.upload_input({},0);candidate.upload_output_gradient({});candidate.forward();candidate.backward(1e-4);
        if(!candidate.download_output().empty()) throw std::runtime_error("nonempty zero-batch output");
        // Overflow must still reject the whole graph update and leave parameters unchanged.
        candidate.upload_parameters(net);
        auto huge=std::vector<double>(xs[0].size(),1e10);
        candidate.upload_input(huge,c.batch);candidate.upload_output_gradient(up);
        bool rejected=false;
        try {candidate.train_step(.01);} catch(const std::overflow_error&) {rejected=true;}
        if(!rejected) throw std::runtime_error("overflow not rejected");
        auto rollback=parameters(candidate.download_parameters());
        set_mode(0,tile);kan::cuda::ResidentNetwork rounded(net,c.batch,precision);
        compare(parameters(rounded.download_parameters()),rollback,"overflow-rollback",0);
        std::printf("check,lifecycle,PASS\n");
        return 0;
    }
    set_mode(mode,tile);kan::cuda::ResidentNetwork r(net,c.batch,precision);
    if(protocol!="resident"&&protocol!="host") throw std::runtime_error("invalid protocol");
    r.upload_input(xs[0],c.batch);r.upload_target(ts[0]);
    int counter=0;
    auto step=[&]{
        const int b=counter++%4;
        if(protocol=="resident") r.train_step(.01,1e-4,kan::cuda::Loss::MeanSquaredError);
        else r.train_step(xs[b],ts[b],c.batch,.01,1e-4);
    };
    if(action=="train") {
        std::printf("mode,tile,precision,protocol,case,branch,seed,step,loss,wall_ms\n");
        const auto start=std::chrono::steady_clock::now();
        for(int j=0;j<steps;++j) {
            step();
            if(j%100==0||j+1==steps) {
                const double loss=r.download_loss();
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                std::printf("%d,%d,%s,%s,%s,%d,%d,%d,%.12g,%.6f\n",mode,tile,precision_name.c_str(),protocol.c_str(),c.name,branch,seed,j+1,loss,ms);
                std::fflush(stdout);
            }
        }
        return 0;
    }
    if(action!="bench") throw std::runtime_error("invalid action");
    for(int j=0;j<std::max(10,steps/10);++j) step();
    r.synchronize();
    std::printf("mode,tile,precision,protocol,case,branch,seed,window,steps,ms_per_step,loss,checksum,allocations\n");
    for(int w=0;w<windows;++w) {
        const auto start=std::chrono::steady_clock::now();
        for(int j=0;j<steps;++j) step();
        r.synchronize();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/steps;
        const double loss=r.download_loss();
        double checksum=0;
        if(w+1==windows) for(double value:parameters(r.download_parameters())) checksum+=value;
        std::printf("%d,%d,%s,%s,%s,%d,%d,%d,%d,%.9f,%.12g,%.12g,%zu\n",mode,tile,precision_name.c_str(),protocol.c_str(),c.name,branch,seed,w,steps,ms,loss,checksum,r.workspace_allocations());
        std::fflush(stdout);
    }
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
