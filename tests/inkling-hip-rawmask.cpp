// Build the real ggml nodes. Inject the already-rounded projection and sink at
// the split boundary, rather than asking CPU arithmetic to predict HIP NaNs.
#include <rocprofiler-sdk-roctx/roctx.h>
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-kperf.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
// Generated from the private tree by build-rawmask-fixture.py; compile the actual
// structural predicates on the host, not a hand-rolled model of their behavior.
#include "rawmask-layout.inc"
static void ck(bool x,const char *s) { if (!x) { fprintf(stderr,"FAIL %s\n",s); exit(1); } }
static int hit[3];
static void logfn(ggml_log_level,const char *s,void *) {
    const char *prefix[]={"INKLING_RAWMASK_HIT ","INKLING_MASKEPI_HIT ","INKLING_GATHER_HIT "};
    for(int j=0;j<3;++j) if(!strncmp(s,prefix[j],strlen(prefix[j]))) ++hit[j];
    fputs(s,stderr);
}
static void test(ggml_backend_t b,FILE *f,bool on,int K,int Q,int S,bool full) {
    const int E=1025,H=24,T=Q*S;
    auto *c=ggml_init({16u<<20,nullptr,true});
    auto input=[&](ggml_type ty,int x,int y,int z,int w){auto*t=ggml_new_tensor_4d(c,ty,x,y,z,w);ggml_set_input(t);return t;};
    auto *weight=input(GGML_TYPE_F32,8,E-1,1,1), *r=input(GGML_TYPE_F32,8,H,T,1);
    auto *p=ggml_mul_mat(c,ggml_pad(c,weight,0,1,0,0),r);
    auto *s0=ggml_mul_mat(c,input(GGML_TYPE_F32,8,H,1,1),input(GGML_TYPE_F32,8,T,1,1));
    auto *s=ggml_reshape_3d(c,s0,1,H,T);
    auto *ids=input(GGML_TYPE_I32,K,T,1,1), *mask=input(GGML_TYPE_F16,K,Q,1,S);
    ggml_set_name(ids,"attn_inp_rel_pos_swa");
    ggml_tensor *n[10];n[0]=ggml_sub(c,p,s);n[1]=ggml_permute(c,n[0],1,0,2,3);n[2]=ggml_cont(c,n[1]);
    n[3]=ggml_get_rows(c,n[2],ids);n[4]=ggml_reshape_4d(c,n[3],H,K,Q,S);n[5]=ggml_permute(c,n[4],2,0,1,3);
    n[6]=ggml_cont(c,n[5]);ggml_set_name(n[6],"kq_b-12");n[7]=ggml_cast(c,mask,GGML_TYPE_F32);
    n[8]=ggml_add(c,n[6],n[7]);n[9]=ggml_cast(c,n[8],GGML_TYPE_F16);ggml_set_name(n[9],"kq_mask_rel_pos-12");ggml_set_output(n[9]);
    auto *plain=ggml_new_graph(c);ggml_build_forward_expand(plain,n[9]);int start=plain->n_nodes-10;
    ck(start>=1 && plain->nodes[start-1]==s,"sink reshape placement");
    for(int j=0;j<10;++j) {
        ck(plain->nodes[start+j]==n[j],"noncontiguous span");
        ck(ggml_node_get_use_count(plain,start+j)==(j==7?2:1),"parent use count");
    }
    ck(s->view_src==s0 && n[1]->view_src==n[0] && n[4]->view_src==n[3] && n[5]->view_src==n[3] &&
       n[7]->src[1]==n[7] && n[9]->src[1]==n[9],"metadata view or CPY self ref");
    ggml_cgraph *tap[9];for(int j=0;j<9;++j){tap[j]=ggml_graph_dup(c,plain,false);ggml_build_forward_expand(tap[j],ggml_dup(c,n[j]));
        ck(ggml_node_get_use_count(tap[j],start+j)==ggml_node_get_use_count(plain,start+j)+1,"outside split use");}
    std::array<int,10> orig_flags;for(int j=0;j<10;++j)orig_flags[j]=n[j]->flags;
    // ggml_set_output also flags every view_src. Restore the whole chain after each case.
    auto restore_flags=[&](){for(int j=0;j<10;++j)n[j]->flags=orig_flags[j];};
    if(!b){bool eligible=T>1;
        ck(inkling_rawmask_layout(plain,start)==eligible,"real layout predicate");
        for(int j=0;j<9;++j){ck(!inkling_rawmask_layout(tap[j],start),"parent consumer not rejected");ggml_set_output(n[j]);ck(!inkling_rawmask_layout(plain,start),"output not rejected");restore_flags();}
        ck(inkling_maskepi_match(plain,start+3),"legacy MASKEPI not eligible");ck(inkling_rawmask_layout(plain,start)==eligible,"restore");
        // Use aligned address offsets in a real host allocation. No device buffer, tensor access, or generic fallback.
        auto *table = n[3]->src[0]; auto *id = n[3]->src[1]; auto *dst = n[6];
        size_t stride=(std::max({ggml_nbytes(table),ggml_nbytes(id),ggml_nbytes(dst)})+63)/64*64;
        std::vector<uint32_t> arena((stride*6+64)/4); uint8_t *mem=(uint8_t*)arena.data();
        for(auto *x : {table,id}) {
            auto *other=(x==id?table:id); other->data=mem+4*stride; dst->data=mem+2*stride;
            auto *z=(uint8_t*)dst->data;
            for(int k=0;k<5;++k) {
                const ptrdiff_t offsets[]={0,-(ptrdiff_t)ggml_nbytes(x)+4,(ptrdiff_t)ggml_nbytes(dst)-4,
                                          -(ptrdiff_t)ggml_nbytes(x),(ptrdiff_t)ggml_nbytes(dst)};
                x->data=z+offsets[k]; ck(inkling_gather_disjoint(n[3],dst)==(k>=3),"short gather alias predicate");
            }
        }
        table->data=nullptr; id->data=nullptr; dst->data=nullptr;
        printf("short GATHER table and ID: exact + both overlapping edges rejected; adjacent edges accepted\n");
        printf("layout-only K=%d Q=%d S=%d prefix=%d; RAW=%d MASKEPI=1; all 9 taps + 9 output flags rejected by actual matcher\n",K,Q,S,start,eligible);ggml_free(c);return;}
    auto *spare=input(GGML_TYPE_F32,int((std::max(ggml_nbytes(n[9]),ggml_nbytes(p))+7)/4+1),1,1,1);
    auto *buf=ggml_backend_alloc_ctx_tensors(c,b);ck(buf,"allocate");
    uint32_t seed=178210;std::vector<uint32_t>a(E*H*T),sink(H*T);std::vector<int32_t>ix(K*T);std::vector<uint16_t>m(K*T);
    constexpr uint32_t special[]={0,0x80000000u,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345,0xff823456,1,0x80000001,0x3f801000,0x3f803000,0x3f800000,0x33800000,0xbf800000,0x477ff000};
    constexpr uint16_t hm[]={0,0x8000,0x7c00,0xfc00,0x7e31,0xfd14,1,0x8001,0x3c00,0xbc00,0x3555,0x7bff,0x8000};
    auto fill=[&](int step){for(size_t i=0;i<a.size();++i){seed=1664525u*seed+1013904223u;a[i]=i%5==0?seed:special[(i+step)%16];}
        for(int t=0;t<T;++t){for(int h=0;h<H;++h)sink[t*H+h]=special[(h+3*t+step)%16];
            for(int k=0;k<K;++k){int pos=t*K+k;ix[pos]=k%19==0?E-1:k%17==0?7:(k%2?(k*37+t*5)%E:(E-1-k%E));
                m[pos]=hm[(k+2*t+step)%13];}}
        // Explicit dual-NaN, signed-zero and two-rounding witnesses. The last
        // is (1 - 2^-26) - 1: baseline F32 SUB rounds to 1 before the add.
        for(int k=1;k<=4;++k)ix[k]=k;
        a[1]=0xff812345; sink[0]=0x7fc54321; m[1]=0xfe12;
        a[E+2]=0x80000000; sink[1]=0; m[2]=0x8000;
        a[2*E+3]=0x3f800000; sink[2]=0x32800000; m[3]=0xbc00;
        a[3*E+4]=0x7f800000; sink[3]=0x7f800000; m[4]=0x7c00;
        ggml_backend_tensor_set(p,a.data(),0,a.size()*4);ggml_backend_tensor_set(s0,sink.data(),0,sink.size()*4);
        ggml_backend_tensor_set(ids,ix.data(),0,ix.size()*4);ggml_backend_tensor_set(mask,m.data(),0,m.size()*2);};
    auto get=[&](ggml_tensor*t){std::vector<uint8_t>v(ggml_nbytes(t));ggml_backend_tensor_get(t,v.data(),0,v.size());return v;};
    auto save=[&](ggml_tensor*t){auto v=get(t);ck(fwrite(v.data(),1,v.size(),f)==v.size(),"write");};
    std::vector<uint8_t> sent(ggml_nbytes(n[0]),0x5a);
    // Codes 0..8 add outside consumers. 10..18 flag the corresponding result.
    // 20..23 partially alias each real read with the output. Code 30 retains a
    // SUB tap through the MASKEPI->GATHER->unfused replay transitions below.
    std::vector<int> phases={-1};if(full){for(int j=0;j<9;++j)phases.insert(phases.end(),{j,-1,10+j,-1});
        phases.insert(phases.end(),{20,21,22,23,-1,30,31,32,31,30,-1});}
    int phase_no=0;bool graphs=!getenv("GGML_CUDA_DISABLE_GRAPHS");
    for(int uidmode=0;uidmode<(full?2:1);++uidmode)for(int ph:phases){
        ggml_cgraph *parent=plain;ggml_tensor *watched=nullptr,*aliased=nullptr;void *old=nullptr,*oldout=n[9]->data;
        if(ph>=0 && ph<9){parent=tap[ph];watched=n[ph];}if(ph>=10 && ph<19){watched=n[ph-10];ggml_set_output(watched);}
        if(ph>=20 && ph<=23){ggml_tensor *x[]={p,s0,ids,mask};aliased=x[ph-20];old=aliased->data;aliased->data=spare->data;
            if(aliased==s0)s->data=aliased->data;n[9]->data=(uint8_t*)spare->data+4;}
        if(ph>=30){ggml_set_output(n[0]);if(ph==31){parent=tap[7];watched=n[7];}if(ph==32){parent=tap[3];watched=n[3];}}
        for(int step=0;step<(full?5:1);++step){
            // Every graph path, including generic GET_ROWS, gets valid row IDs.
            fill(step);ggml_backend_tensor_set(n[0],sent.data(),0,sent.size());
            if(watched && watched->view_src!=n[0] && watched!=n[0]){std::vector<uint8_t>bad(ggml_nbytes(watched),0x5a);ggml_backend_tensor_set(watched,bad.data(),0,bad.size());}
            ggml_cgraph g=*parent;g.nodes+=start;g.n_nodes=10;g.uid=uidmode?0:703+K+(aliased?1000*ph:0); // a new allocation layout is a new scheduler uid
g.size=0;g.n_leafs=0;g.leafs=nullptr;g.grads=nullptr;g.grad_accs=nullptr;
            std::array<int,3> prev={hit[0],hit[1],hit[2]};std::string tag="rawmask_K"+std::to_string(K)+"_uid"+std::to_string(uidmode)+"_phase"+std::to_string(phase_no)+"_case"+std::to_string(ph)+"_step"+std::to_string(step);
            roctxRangePushA(tag.c_str());ck(ggml_backend_graph_compute(b,&g)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(b);roctxRangePop();
            bool longhit=on && ph<0;ck((get(n[0])==sent)==longhit,"stale or unexpectedly materialized SUB");
            if(!longhit)ck(prev[0]==hit[0],"rejected site fused");
            if(graphs && step>=2)for(int j=0;j<3;++j)ck(prev[j]==hit[j],"expected replay, observed eager dispatch");
            if(on && ph>=30 && step<2 && (prev[1]!=hit[1]||prev[2]!=hit[2])){ck(ph!=32 && (ph==30?prev[1]!=hit[1]:prev[2]!=hit[2]),"wrong shorter fallback");}
            auto bytes=get(n[9]);uint16_t z,cancel;memcpy(&z,bytes.data()+2*(K*Q+2),2);memcpy(&cancel,bytes.data()+2*(2*K*Q+3),2);
            ck(z==0x8000 && cancel==0,"signed zero or separate SUB rounding");ck(fwrite(bytes.data(),1,bytes.size(),f)==bytes.size(),"write");if(watched)save(watched);if(ph>=30)save(n[0]);
        }
        ++phase_no;
        if(aliased){aliased->data=old;if(aliased==s0)s->data=old;n[9]->data=oldout;}
        restore_flags();
    }
    ggml_backend_buffer_free(buf);ggml_free(c);
}
int main(int argc,char**argv){
    if(argc==2 && !strcmp(argv[1],"--layout")){test(nullptr,nullptr,false,65,3,2,true);test(nullptr,nullptr,false,256,256,1,false);test(nullptr,nullptr,false,65,1,1,false);return 0;}
    // the OFF arm runs with GGML_KPERF_OFF=inkling.rawmask, MASKEPI and GATHER stay on
    if(argc!=3)return 2;setenv("GGML_KPERF_TRACE","inkling.rawmask",1);bool on=ggml_kperf_on(GGML_KPERF_INKLING_RAWMASK);
    ck(ggml_kperf_on(GGML_KPERF_INKLING_MASKEPI) && ggml_kperf_on(GGML_KPERF_INKLING_GATHER),"inkling.maskepi and inkling.gather must be on");
    ggml_log_set(logfn,nullptr);ggml_backend_load_all_from_path(argv[1]);auto*b=ggml_backend_init_by_name("ROCm0",nullptr);ck(b,"GPU");
    std::ifstream maps("/proc/self/maps");bool hip=false;std::string l;while(std::getline(maps,l)){auto i=l.find('/');if(i==l.npos)continue;auto p=l.substr(i),name=p.substr(p.rfind('/')+1);if(name.rfind("libggml",0))continue;
        ck(p.rfind(std::string(argv[1])+"/",0)==0,"mixed ggml mappings");hip|=!name.rfind("libggml-hip.so",0);}ck(hip,"HIP unmapped");
    auto*f=fopen(argv[2],"wbx");ck(f,"exclusive output");test(b,f,on,65,3,2,true);test(b,f,on,256,256,1,false);ck((hit[0]>0)==on,"hit count");ck(!fclose(f),"close");ggml_backend_free(b);
}
