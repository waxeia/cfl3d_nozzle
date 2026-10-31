#define NOZZLE_INLINE __host__ __device__ inline
#include "explicit_backend_common.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <deque>
#include <iostream>

static void cudaCheck(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}

__device__ double atomicMinDouble(double*address,double val){auto p=reinterpret_cast<unsigned long long*>(address);unsigned long long old=*p,assumed;while(val<__longlong_as_double(old)){assumed=old;old=atomicCAS(p,assumed,__double_as_longlong(val));if(old==assumed)break;}return __longlong_as_double(old);}

__global__ void faceKernel(const double*U,const DeviceFace*faces,int nf,DeviceParams p,double*flux){
  int fi=blockIdx.x*blockDim.x+threadIdx.x;if(fi>=nf)return;DeviceFace f=faces[fi];double L[4],R[4];for(int k=0;k<4;k++)L[k]=U[4*f.left+k];if(f.right>=0)for(int k=0;k<4;k++)R[k]=U[4*f.right+k];else dGhost(f,L,p,R);dRoe(L,R,f.nx,f.ny,p.gamma,&flux[4*fi]);
}
__global__ void residualKernel(const DeviceFace*faces,const DeviceCell*cells,const double*flux,int nc,double*residual){
  int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nc)return;for(int k=0;k<4;k++){double s=0;for(int m=0;m<4;m++){int fi=cells[n].face[m];double sign=faces[fi].left==n?1:-1;s+=sign*flux[4*fi+k]*faces[fi].length;}residual[4*n+k]=s;}
}
__global__ void diagnosticKernel(const double*U,const double*residual,const DeviceCell*cells,int nc,double gamma,double*diag){
  int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nc)return;double s=0;for(int k=0;k<4;k++){double z=residual[4*n+k]/cells[n].area;s+=z*z;}atomicAdd(&diag[0],s);DPrim q=dPrimitive(&U[4*n],gamma);atomicMinDouble(&diag[9],q.rho);atomicMinDouble(&diag[10],q.p);
}
__global__ void boundaryDiagnosticKernel(const double*U,const DeviceFace*faces,const double*flux,int nf,double gamma,double*diag){
  int fi=blockIdx.x*blockDim.x+threadIdx.x;if(fi>=nf)return;DeviceFace f=faces[fi];if(f.bc==1)atomicAdd(&diag[1],flux[4*fi]*f.length);if(f.bc==2){atomicAdd(&diag[2],flux[4*fi]*f.length);DPrim q=dPrimitive(&U[4*f.left],gamma);atomicAdd(&diag[3],q.rho*f.length);atomicAdd(&diag[4],q.u*f.length);atomicAdd(&diag[5],q.v*f.length);atomicAdd(&diag[6],q.p*f.length);atomicAdd(&diag[7],f.length);}
}
__global__ void trialKernel(const double*U,const double*residual,const DeviceFace*faces,const DeviceCell*cells,int nc,DeviceParams p,double alpha,double*trial,int*invalid){
  int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nc)return;DPrim q=dPrimitive(&U[4*n],p.gamma);double sum=0;for(int m=0;m<4;m++){DeviceFace f=faces[cells[n].face[m]];sum+=dWave(q,f.nx,f.ny,p.gamma)*f.length;}double dt=p.cfl*cells[n].area/fmax(sum,1e-12);double v[4];for(int k=0;k<4;k++){v[k]=U[4*n+k]-alpha*dt*residual[4*n+k]/cells[n].area;trial[4*n+k]=v[k];}DPrim qt=dPrimitive(v,p.gamma);if(!isfinite(qt.rho)||!isfinite(qt.p)||qt.rho<=1e-10||qt.p<=1e-10)atomicExch(invalid,1);
}

struct Sample {double mass,rho,u,v,p;};
static double sampleChange(const Sample&a,const Sample&b,double c0){return std::max({backendRelative(a.mass,b.mass),backendRelative(a.rho,b.rho),backendRelative(a.u,b.u),std::abs(a.v-b.v)/c0,backendRelative(a.p,b.p)});}

int main(int argc,char**argv){
  try{
    if(argc<2)throw std::runtime_error("usage: nozzle_cuda CONFIG [OUTPUT_DIR]");BackendConfig c=loadBackendConfig(argv[1]);if(argc>2)c.output=argv[2];BackendMesh mesh;mesh.build(c);if(!mesh.geometryOk())throw std::runtime_error("geometry check failed");auto hostU=backendInitialize(c,mesh);std::filesystem::create_directories(c.output);std::ofstream hist(c.output+"/residual.csv");hist<<std::setprecision(17)<<"iter,residual,normalized,inlet_mass,outlet_mass,mass_imbalance,window_change,min_rho,min_p\n";
    int nc=static_cast<int>(mesh.cells.size()),nf=static_cast<int>(mesh.faces.size());size_t us=4ull*nc*sizeof(double),fs=4ull*nf*sizeof(double);double *U=nullptr,*trial=nullptr,*residual=nullptr,*flux=nullptr,*diag=nullptr;DeviceFace*dFaces=nullptr;DeviceCell*dCells=nullptr;int*invalid=nullptr;
    auto transfer0=std::chrono::steady_clock::now();cudaCheck(cudaMalloc(&U,us),"cudaMalloc U");cudaCheck(cudaMalloc(&trial,us),"cudaMalloc trial");cudaCheck(cudaMalloc(&residual,us),"cudaMalloc residual");cudaCheck(cudaMalloc(&flux,fs),"cudaMalloc flux");cudaCheck(cudaMalloc(&dFaces,nf*sizeof(DeviceFace)),"cudaMalloc faces");cudaCheck(cudaMalloc(&dCells,nc*sizeof(DeviceCell)),"cudaMalloc cells");cudaCheck(cudaMalloc(&diag,11*sizeof(double)),"cudaMalloc diag");cudaCheck(cudaMalloc(&invalid,sizeof(int)),"cudaMalloc invalid");cudaCheck(cudaMemcpy(U,hostU.data(),us,cudaMemcpyHostToDevice),"copy U");cudaCheck(cudaMemcpy(dFaces,mesh.faces.data(),nf*sizeof(DeviceFace),cudaMemcpyHostToDevice),"copy faces");cudaCheck(cudaMemcpy(dCells,mesh.cells.data(),nc*sizeof(DeviceCell),cudaMemcpyHostToDevice),"copy cells");cudaCheck(cudaDeviceSynchronize(),"initial sync");double transferSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-transfer0).count();
    DeviceParams p=deviceParams(c);int threads=256,bf=(nf+threads-1)/threads,bc=(nc+threads-1)/threads,status=2,lastIter=0;double ref=0,finalNorm=0,imbalance=INFINITY,stability=INFINITY;std::deque<Sample>window;auto start=std::chrono::steady_clock::now();
    for(int it=0;it<c.max_iter;it++){
      lastIter=it;faceKernel<<<bf,threads>>>(U,dFaces,nf,p,flux);residualKernel<<<bc,threads>>>(dFaces,dCells,flux,nc,residual);
      double hd[11]={0,0,0,0,0,0,0,0,0,std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};cudaCheck(cudaMemcpy(diag,hd,sizeof(hd),cudaMemcpyHostToDevice),"clear diagnostics");diagnosticKernel<<<bc,threads>>>(U,residual,dCells,nc,c.gamma,diag);boundaryDiagnosticKernel<<<bf,threads>>>(U,dFaces,flux,nf,c.gamma,diag);cudaCheck(cudaMemcpy(hd,diag,sizeof(hd),cudaMemcpyDeviceToHost),"copy diagnostics");
      double rn=std::sqrt(hd[0]/nc);if(it==0)ref=std::max(rn,1e-12);finalNorm=rn/ref;imbalance=std::abs(hd[1]+hd[2])/std::max(1e-12,std::abs(hd[1]));Sample sample{hd[2],hd[3]/hd[7],hd[4]/hd[7],hd[5]/hd[7],hd[6]/hd[7]};window.push_back(sample);if(static_cast<int>(window.size())>c.stability_window)window.pop_front();stability=static_cast<int>(window.size())==c.stability_window?sampleChange(window.front(),window.back(),std::sqrt(c.gamma*c.gas_R*c.T0)):INFINITY;hist<<it<<","<<rn<<","<<finalNorm<<","<<hd[1]<<","<<hd[2]<<","<<imbalance<<","<<stability<<","<<hd[9]<<","<<hd[10]<<"\n";if(it%c.report_every==0)std::cout<<"iter "<<it<<" residual "<<finalNorm<<" mass "<<hd[1]<<" "<<hd[2]<<" stability "<<stability<<"\n";
      if(finalNorm<=c.residual_drop&&imbalance<=c.mass_tolerance&&stability<=c.stability_tolerance){status=0;break;}
      double alpha=1;bool accepted=false;for(int retry=0;retry<12;retry++){cudaCheck(cudaMemset(invalid,0,sizeof(int)),"clear invalid");trialKernel<<<bc,threads>>>(U,residual,dFaces,dCells,nc,p,alpha,trial,invalid);int bad=0;cudaCheck(cudaMemcpy(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost),"copy invalid");if(!bad){std::swap(U,trial);accepted=true;break;}alpha*=.5;}if(!accepted){status=3;break;}
    }
    cudaCheck(cudaDeviceSynchronize(),"final sync");double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto transfer1=std::chrono::steady_clock::now();cudaCheck(cudaMemcpy(hostU.data(),U,us,cudaMemcpyDeviceToHost),"copy final field");cudaCheck(cudaDeviceSynchronize(),"final copy sync");transferSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-transfer1).count();backendWriteField(c.output,c,mesh,hostU);std::ofstream meta(c.output+"/run_status.txt");meta<<std::setprecision(17)<<"status="<<(status==0?"converged":status==2?"max_iter":"invalid_state")<<"\nseconds="<<seconds<<"\ntransfer_seconds="<<transferSeconds<<"\niterations="<<lastIter+1<<"\nsolver=explicit\nbackend=cuda\nflux=roe\nreconstruction=first\nnormalized_residual="<<finalNorm<<"\nmass_imbalance="<<imbalance<<"\nwindow_change="<<stability<<"\n";
    cudaFree(U);cudaFree(trial);cudaFree(residual);cudaFree(flux);cudaFree(dFaces);cudaFree(dCells);cudaFree(diag);cudaFree(invalid);return status;
  }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 5;}
}
