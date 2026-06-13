"""
BÀI 7 – TỐI ƯU ĐA MỤC TIÊU PARETO (NSGA-II xấp xỉ bằng weighted-sum scan)
4 mục tiêu: f1=GDP gain, f2=bất bình đẳng, f3=phát thải, f4=rủi ro an ninh
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from mpl_toolkits.mplot3d import Axes3D

print("="*60); print("BÀI 7 – TỐI ƯU ĐA MỤC TIÊU (Pareto front)"); print("="*60)

R,J,N=6,4,24
beta=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],
               [1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
e  =np.array([0.42,0.55,0.48,0.32,0.62,0.38])
rho=np.array([0.18,0.45,0.28,0.12,0.52,0.22])
sig=np.array([0.32,0.28,0.30,0.35,0.25,0.30])

def f1(X): return (beta*X).sum()
def f2(X): s=X.sum(axis=1); return np.abs(s-s.mean()).mean()
def f3(X): return (e*(X[:,0]+X[:,2])).sum()
def f4(X): return (rho*X[:,2]).sum()-(sig*X[:,3]).sum()

# Ràng buộc cơ bản dùng trong LP
def build_lp_constraints():
    Al,bl=[],[]
    Al.append(np.ones(N));  bl.append(50000)
    for r in range(R):
        row=np.zeros(N); row[r*J:(r+1)*J]=-1; Al.append(row); bl.append(-5000)
    for r in range(R):
        row=np.zeros(N); row[r*J:(r+1)*J]=1; Al.append(row); bl.append(12000)
    row=np.zeros(N)
    for r in range(R): row[r*J+3]=-1
    Al.append(row); bl.append(-12000)
    return np.array(Al),np.array(bl)

A_ub,b_ub=build_lp_constraints()

# Weighted-sum scalarization để quét Pareto front
# w1*(-f1) + w2*f2 + w3*f3 + w4*f4 → minimize
# Dùng scipy.optimize.minimize SLSQP vì f2,f3,f4 nonlinear với x
pareto_F, pareto_X = [], []
np.random.seed(42)
n_samples=200
print("Đang quét Pareto front (200 nghiệm)...")

for _ in range(n_samples):
    w=np.random.dirichlet([1,1,1,1])  # random weights
    def obj(x):
        X=x.reshape(R,J)
        return w[0]*(-f1(X)) + w[1]*f2(X)/1000 + w[2]*f3(X)/1000 + w[3]*f4(X)/1000
    def grad(x):
        return None  # let optimizer estimate

    x0=np.random.uniform(0,5000,N)
    # Normalize x0 to satisfy budget
    x0=x0/x0.sum()*45000

    cons=[
        {'type':'ineq','fun':lambda x: 50000-x.sum()},
        {'type':'ineq','fun':lambda x: x.sum()-30000},
    ]
    for r in range(R):
        cons.append({'type':'ineq','fun':lambda x,r=r: x[r*J:(r+1)*J].sum()-5000})
        cons.append({'type':'ineq','fun':lambda x,r=r: 12000-x[r*J:(r+1)*J].sum()})
    cons.append({'type':'ineq','fun':lambda x: x[3::J].sum()-12000})

    bounds_slsqp=[(0,12000)]*N
    res=minimize(obj,x0,method='SLSQP',bounds=bounds_slsqp,constraints=cons,
                 options={'maxiter':100,'ftol':1e-4})
    if res.success:
        X=res.x.reshape(R,J)
        pareto_F.append([f1(X),-f2(X),f3(X),f4(X)])
        pareto_X.append(res.x)

pareto_F=np.array(pareto_F)
print(f"Thu được {len(pareto_F)} nghiệm hợp lệ")

# CÂU 7.4.3 – TOPSIS trên Pareto front
def topsis_pareto(F, w, is_b):
    Fn=F/np.sqrt((F**2).sum(axis=0)+1e-10)
    V=Fn*w
    Ap=np.where(is_b,V.max(axis=0),V.min(axis=0))
    An=np.where(is_b,V.min(axis=0),V.max(axis=0))
    Sp=np.sqrt(((V-Ap)**2).sum(axis=1))
    Sn=np.sqrt(((V-An)**2).sum(axis=1))
    return Sn/(Sp+Sn)

F_topsis=np.column_stack([pareto_F[:,0],-pareto_F[:,1],-pareto_F[:,2],-pareto_F[:,3]])
w_policy=np.array([0.40,0.25,0.20,0.15])
is_b=[True,True,True,True]  # sau khi đổi dấu, tất cả đều "lớn hơn = tốt hơn"
C_star=topsis_pareto(F_topsis,w_policy,is_b)
best_idx=np.argmax(C_star)
X_best=pareto_X[best_idx].reshape(R,J)

print(f"\nCÂU 7.4.3 – Nghiệm thỏa hiệp TOPSIS (w={w_policy}):")
regs=['TDMN-PB','DBSH','BTB-DHMT','TN','DNB','DBSCL']
df_best=pd.DataFrame(X_best.round(0),index=regs,columns=['I','D','AI','H'])
print(df_best.to_string())
print(f"\nf1 GDP gain = {f1(X_best):,.0f} tỷ")
print(f"f2 Bất bình đẳng = {f2(X_best):,.2f}")
print(f"f3 Phát thải = {f3(X_best):,.2f}")
print(f"f4 Rủi ro an ninh = {f4(X_best):,.2f}")

# CÂU 7.4.4 – Chi phí cơ hội
best_f1_idx=np.argmax(pareto_F[:,0])
X_maxgdp=pareto_X[best_f1_idx].reshape(R,J)
print(f"\nCÂU 7.4.4 – Nghiệm GDP cao nhất vs Thỏa hiệp:")
print(f"  GDP max:     f1={f1(X_maxgdp):,.0f}  f2={f2(X_maxgdp):,.0f}  f3={f3(X_maxgdp):,.0f}")
print(f"  Thỏa hiệp:  f1={f1(X_best):,.0f}  f2={f2(X_best):,.0f}  f3={f3(X_best):,.0f}")
print(f"  Hi sinh GDP: {(f1(X_maxgdp)-f1(X_best))/f1(X_maxgdp)*100:.1f}%  "+
      f"Giảm BĐ: {(f2(X_maxgdp)-f2(X_best))/f2(X_maxgdp)*100:.1f}%  "+
      f"Giảm phát thải: {(f3(X_maxgdp)-f3(X_best))/f3(X_maxgdp)*100:.1f}%")

# Biểu đồ
fig=plt.figure(figsize=(18,6))
fig.suptitle('Bài 7 – Pareto Front: 4 mục tiêu phát triển kinh tế số VN',fontweight='bold')

ax1=fig.add_subplot(131,projection='3d')
sc=ax1.scatter(-pareto_F[:,0]/1000, pareto_F[:,2], -pareto_F[:,1],
               c=C_star, cmap='RdYlGn', s=15, alpha=0.6)
ax1.scatter(-pareto_F[best_idx,0]/1000, pareto_F[best_idx,2], -pareto_F[best_idx,1],
            color='red', s=100, marker='*', label='Thỏa hiệp')
ax1.set_xlabel('GDP gain (k tỷ)'); ax1.set_ylabel('Phát thải'); ax1.set_zlabel('Bình đẳng')
ax1.set_title('Pareto 3D\n(f1, f3, f2)',fontweight='bold'); ax1.legend()
plt.colorbar(sc,ax=ax1,label='C* TOPSIS',shrink=0.5)

ax2=fig.add_subplot(132)
ax2.scatter(-pareto_F[:,0]/1000, pareto_F[:,2], c=pareto_F[:,3], cmap='RdYlBu_r', alpha=0.5, s=20)
ax2.scatter(-pareto_F[best_idx,0]/1000, pareto_F[best_idx,2], color='red', s=150, marker='*', zorder=5)
ax2.set_xlabel('GDP gain (k tỷ)'); ax2.set_ylabel('Phát thải CO2')
ax2.set_title('Đánh đổi GDP ↔ Phát thải\n(màu=rủi ro an ninh)',fontweight='bold')
ax2.grid(alpha=0.3)

ax3=fig.add_subplot(133)
F_norm=(F_topsis-F_topsis.min(axis=0))/(F_topsis.max(axis=0)-F_topsis.min(axis=0)+1e-10)
for i in range(min(50,len(pareto_F))):
    alpha=0.3 if i!=best_idx else 1.0
    color='red' if i==best_idx else 'steelblue'
    lw=2.5 if i==best_idx else 0.5
    ax3.plot(range(4), F_norm[i], color=color, alpha=alpha, linewidth=lw)
ax3.set_xticks(range(4)); ax3.set_xticklabels(['GDP','Bình đẳng','Môi trường','An ninh'])
ax3.set_ylabel('Giá trị chuẩn hóa [0,1]')
ax3.set_title('Parallel Coordinates\n(đỏ=nghiệm thỏa hiệp)',fontweight='bold'); ax3.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('bai07_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()
print("""
THẢO LUẬN:
a) Đánh đổi GDP-bao trùm: nghiệm GDP cao nhất thường phân bổ vốn tập trung DNB/DBSH
   → bất bình đẳng vùng tăng rõ rệt (f2 tăng) → phản ánh cơ cấu hai tốc độ của kinh tế VN

b) Trọng số (0.40;0.25;0.20;0.15) ưu tiên tăng trưởng; để phù hợp COP26 nên tăng w_env lên 0.25+
   Để phù hợp NQ57: giữ w_GDP cao nhưng tăng w_security (an ninh mạng là ưu tiên 2024)

c) NSGA-II khác LP: LP cho 1 nghiệm, NSGA-II cho tập nghiệm Pareto
   → Không thay thế quyết định chính trị; cung cấp 'menu' các đánh đổi cho nhà hoạch định lựa chọn
""")
print("✓ Bài 7 → bai07_ket_qua.png")
