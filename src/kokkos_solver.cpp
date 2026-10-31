#include <Kokkos_Core.hpp>
#define NOZZLE_INLINE KOKKOS_INLINE_FUNCTION
#include "explicit_backend_common.hpp"

#include <chrono>
#include <deque>
#include <iostream>

using StateView=Kokkos::View<double**,Kokkos::LayoutRight>;
struct KSample {double mass,rho,u,v,p;};
static double kChange(const KSample&a,const KSample&b,double c0){return std::max({backendRelative(a.mass,b.mass),backendRelative(a.rho,b.rho),backendRelative(a.u,b.u),std::abs(a.v-b.v)/c0,backendRelative(a.p,b.p)});}

int main(int argc,char**argv){
  if(argc<2){std::cerr<<"usage: nozzle_kokkos CONFIG [OUTPUT_DIR]\n";return 5;}
  int result=5;Kokkos::initialize();
  try{
    BackendConfig c=loadBackendConfig(argv[1]);if(argc>2)c.output=argv[2];BackendMesh mesh;mesh.build(c);if(!mesh.geometryOk())throw std::runtime_error("geometry check failed");auto initial=backendInitialize(c,mesh);std::filesystem::create_directories(c.output);std::ofstream hist(c.output+"/residual.csv");hist<<std::setprecision(17)<<"iter,residual,normalized,inlet_mass,outlet_mass,mass_imbalance,window_change,min_rho,min_p\n";int nc=static_cast<int>(mesh.cells.size()),nf=static_cast<int>(mesh.faces.size());
    StateView U("U",nc,4),trial("trial",nc,4),residual("residual",nc,4),flux("flux",nf,4);Kokkos::View<DeviceFace*>faces("faces",nf);Kokkos::View<DeviceCell*>cells("cells",nc);auto hU=Kokkos::create_mirror_view(U);auto hFaces=Kokkos::create_mirror_view(faces);auto hCells=Kokkos::create_mirror_view(cells);for(int n=0;n<nc;n++)for(int k=0;k<4;k++)hU(n,k)=initial[4*n+k];for(int i=0;i<nf;i++)hFaces(i)=mesh.faces[i];for(int i=0;i<nc;i++)hCells(i)=mesh.cells[i];auto transfer0=std::chrono::steady_clock::now();Kokkos::deep_copy(U,hU);Kokkos::deep_copy(faces,hFaces);Kokkos::deep_copy(cells,hCells);Kokkos::fence();double transferSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-transfer0).count();
    DeviceParams p=deviceParams(c);Kokkos::View<double*>diag("diag",8);int status=2,lastIter=0;double ref=0,finalNorm=0,imbalance=INFINITY,stability=INFINITY;std::deque<KSample>window;auto start=std::chrono::steady_clock::now();
    for(int it=0;it<c.max_iter;it++){
      lastIter=it;
      Kokkos::parallel_for("face_flux",nf,KOKKOS_LAMBDA(int fi){DeviceFace f=faces(fi);double L[4],R[4],F[4];for(int k=0;k<4;k++)L[k]=U(f.left,k);if(f.right>=0)for(int k=0;k<4;k++)R[k]=U(f.right,k);else dGhost(f,L,p,R);dRoe(L,R,f.nx,f.ny,p.gamma,F);for(int k=0;k<4;k++)flux(fi,k)=F[k];});
      Kokkos::parallel_for("cell_residual",nc,KOKKOS_LAMBDA(int n){for(int k=0;k<4;k++){double s=0;for(int m=0;m<4;m++){int fi=cells(n).face[m];double sign=faces(fi).left==n?1:-1;s+=sign*flux(fi,k)*faces(fi).length;}residual(n,k)=s;}});
      double sumsq=0,minrho=0,minp=0;Kokkos::parallel_reduce("residual_norm",nc,KOKKOS_LAMBDA(int n,double&s){for(int k=0;k<4;k++){double z=residual(n,k)/cells(n).area;s+=z*z;}},sumsq);Kokkos::parallel_reduce("min_rho",nc,KOKKOS_LAMBDA(int n,double&v){DPrim q=dPrimitive(&U(n,0),p.gamma);if(q.rho<v)v=q.rho;},Kokkos::Min<double>(minrho));Kokkos::parallel_reduce("min_p",nc,KOKKOS_LAMBDA(int n,double&v){DPrim q=dPrimitive(&U(n,0),p.gamma);if(q.p<v)v=q.p;},Kokkos::Min<double>(minp));Kokkos::deep_copy(diag,0.0);
      Kokkos::parallel_for("boundary_diag",nf,KOKKOS_LAMBDA(int fi){DeviceFace f=faces(fi);if(f.bc==1)Kokkos::atomic_add(&diag(0),flux(fi,0)*f.length);if(f.bc==2){Kokkos::atomic_add(&diag(1),flux(fi,0)*f.length);DPrim q=dPrimitive(&U(f.left,0),p.gamma);Kokkos::atomic_add(&diag(2),q.rho*f.length);Kokkos::atomic_add(&diag(3),q.u*f.length);Kokkos::atomic_add(&diag(4),q.v*f.length);Kokkos::atomic_add(&diag(5),q.p*f.length);Kokkos::atomic_add(&diag(6),f.length);}});auto hd=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),diag);
      double rn=std::sqrt(sumsq/nc);if(it==0)ref=std::max(rn,1e-12);finalNorm=rn/ref;imbalance=std::abs(hd(0)+hd(1))/std::max(1e-12,std::abs(hd(0)));KSample sample{hd(1),hd(2)/hd(6),hd(3)/hd(6),hd(4)/hd(6),hd(5)/hd(6)};window.push_back(sample);if(static_cast<int>(window.size())>c.stability_window)window.pop_front();stability=static_cast<int>(window.size())==c.stability_window?kChange(window.front(),window.back(),std::sqrt(c.gamma*c.gas_R*c.T0)):INFINITY;hist<<it<<","<<rn<<","<<finalNorm<<","<<hd(0)<<","<<hd(1)<<","<<imbalance<<","<<stability<<","<<minrho<<","<<minp<<"\n";if(it%c.report_every==0)std::cout<<"iter "<<it<<" residual "<<finalNorm<<" mass "<<hd(0)<<" "<<hd(1)<<" stability "<<stability<<"\n";
      if(finalNorm<=c.residual_drop&&imbalance<=c.mass_tolerance&&stability<=c.stability_tolerance){status=0;break;}
      double alpha=1;bool accepted=false;for(int retry=0;retry<12;retry++){
        Kokkos::parallel_for("trial",nc,KOKKOS_LAMBDA(int n){DPrim q=dPrimitive(&U(n,0),p.gamma);double sum=0;for(int m=0;m<4;m++){DeviceFace f=faces(cells(n).face[m]);sum+=dWave(q,f.nx,f.ny,p.gamma)*f.length;}double dt=p.cfl*cells(n).area/fmax(sum,1e-12);for(int k=0;k<4;k++)trial(n,k)=U(n,k)-alpha*dt*residual(n,k)/cells(n).area;});
        int invalid=0;Kokkos::parallel_reduce("validate_trial",nc,KOKKOS_LAMBDA(int n,int&bad){DPrim qt=dPrimitive(&trial(n,0),p.gamma);int here=(!Kokkos::isfinite(qt.rho)||!Kokkos::isfinite(qt.p)||qt.rho<=1e-10||qt.p<=1e-10)?1:0;if(here>bad)bad=here;},Kokkos::Max<int>(invalid));
        if(!invalid){std::swap(U,trial);accepted=true;break;}alpha*=.5;
      }if(!accepted){status=3;break;}
    }
    Kokkos::fence();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto transfer1=std::chrono::steady_clock::now();hU=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),U);Kokkos::fence();transferSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-transfer1).count();for(int n=0;n<nc;n++)for(int k=0;k<4;k++)initial[4*n+k]=hU(n,k);backendWriteField(c.output,c,mesh,initial);std::string execution=Kokkos::DefaultExecutionSpace::name(),backend="kokkos-";std::transform(execution.begin(),execution.end(),std::back_inserter(backend),[](unsigned char x){return static_cast<char>(std::tolower(x));});std::ofstream meta(c.output+"/run_status.txt");meta<<std::setprecision(17)<<"status="<<(status==0?"converged":status==2?"max_iter":"invalid_state")<<"\nseconds="<<seconds<<"\ntransfer_seconds="<<transferSeconds<<"\niterations="<<lastIter+1<<"\nsolver=explicit\nbackend="<<backend<<"\nexecution_space="<<execution<<"\nflux=roe\nreconstruction=first\nnormalized_residual="<<finalNorm<<"\nmass_imbalance="<<imbalance<<"\nwindow_change="<<stability<<"\n";result=status;
  }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";result=5;}
  Kokkos::finalize();return result;
}
