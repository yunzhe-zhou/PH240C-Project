from algorithm_new import *

np.random.seed(8)
d = 12
W = generate_W(d=d, prob=0.5)
N=20
T=200
size=int(N*T)
arparams = np.r_[1,-0.5]
maparam = np.r_[1]
xs = np.zeros((size, d))
for i in range(d):
    for n in range(N):
        if n==0:
            y=arma_generate_sample(arparams, maparam, T + 2000)
            y=y[2000:(T+2000)]
        else:
            q=arma_generate_sample(arparams, maparam, T + 2000)
            put=copy.deepcopy(q[2000:(T+2000)])
            y=np.concatenate((y, put), axis=0)
    xs[:, i] = y + xs.dot(W[i, :])
# standardize xs
for m in range(d):
    xs[:,m]=(xs[:,m]-np.mean(xs[:,m]))/(np.std(xs[:,m]))
b_all=[copy.deepcopy(W),copy.deepcopy(W)]
root_all=root_relationship_all(b_all)

j=11
result_ls=[]
for k in range(11):
    range_K=[5,10,20]
    result=cal_infer(j=j,k=k,root_all=root_all,range_K=range_K,b_all=b_all,xs=xs,d=d,M=1000,B1=100,B2=10,J=1000,N=N,T=T,version=2)
    print("null value: ",W[j,k])
    print("p value: ",result)
    result_ls.append(result)
    print("p value all: ",result_ls)
    np.save("data",result_ls)