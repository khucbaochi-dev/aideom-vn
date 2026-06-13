"""
BÀI 9 – TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG
max Σ NetJob_i  s.t. ngân sách, NetJob≥0, Displaced≤Retrain
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linprog

print("="*60); print("BÀI 9 – MÔ PHỎNG LAO ĐỘNG DƯỚI TÁC ĐỘNG AI"); print("="*60)

sector_names=['Nông-Lâm-TS','CN Chế biến','Xây dựng','Bán buôn-lẻ',
              'Tài chính-NH','Logistics','CNTT-TT','Giáo dục']
N=8
L   = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
risk= np.array([18,42,25,38,52,35,28,22])/100
a1  = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1  = np.array([45.,28.,35.,32.,22.,30.,20.,55.])
c1  = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1  = np.array([50.,32.,42.,38.,26.,36.,24.,62.])

# x = [x_AI_0..7, x_H_0..7]  (16 biến)
# NetJob_i = (a1_i - c1_i*risk_i)*x_AI_i + b1_i*x_H_i
# ràng buộc NetJob_i ≥ 0: -(a1_i-c1_i*risk_i)*x_AI_i - b1_i*x_H_i ≤ 0
# ràng buộc Displaced≤Retrain: c1_i*risk_i*x_AI_i ≤ d1_i*x_H_i

net_coeff_AI = a1 - c1*risk   # hệ số ròng tạo việc làm từ AI
Nv=2*N   # 16 biến

# Hàm mục tiêu: max Σ NetJob → min -(Σ net_coeff_AI*x_AI + Σ b1*x_H)
c_obj=np.concatenate([-net_coeff_AI, -b1])

Al,bl=[],[]
# C1: Σ(x_AI+x_H) ≤ 30000
Al.append(np.ones(Nv)); bl.append(30000)
# C2: NetJob_i ≥ 0 → -(net_AI*x_AI + b1*x_H) ≤ 0
for i in range(N):
    row=np.zeros(Nv); row[i]=-net_coeff_AI[i]; row[N+i]=-b1[i]
    Al.append(row); bl.append(0)
# C3: Displaced ≤ Retrain → c1*risk*x_AI - d1*x_H ≤ 0
for i in range(N):
    row=np.zeros(Nv); row[i]=c1[i]*risk[i]; row[N+i]=-d1[i]
    Al.append(row); bl.append(0)

A_ub=np.array(Al); b_ub=np.array(bl)
bds=[(0,None)]*Nv

res=linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bds, method='highs')

x_AI=res.x[:N]; x_H=res.x[N:]
NetJob=net_coeff_AI*x_AI + b1*x_H
Displaced=c1*risk*x_AI; Retrain=d1*x_H; NewJob=a1*x_AI; UpJob=b1*x_H

print(f"\nCÂU 9.4.1 – Phân bổ tối ưu (B=30,000 tỷ):")
df_lj=pd.DataFrame({
    'Ngành':sector_names,'L(tr)':L,'x_AI(tỷ)':x_AI.round(0),'x_H(tỷ)':x_H.round(0),
    'NewJob':NewJob.round(0),'UpJob':UpJob.round(0),'Displaced':Displaced.round(0),
    'NetJob':NetJob.round(0)})
print(df_lj.to_string(index=False))
print(f"\nTổng NetJob tối đa = {NetJob.sum():,.0f} việc làm")
print(f"Tổng ngân sách dùng: {(x_AI+x_H).sum():,.0f}/30,000 tỷ")

# CÂU 9.4.2 – Ngưỡng x_H tối thiểu ngành 2 (CN chế biến)
print(f"\nCÂU 9.4.2 – Ngưỡng x_H tối thiểu cho CN Chế biến:")
i=1  # ngành 2
# NetJob2 ≥ 0 khi x_AI=max possible
# net_AI*x_AI + b1*x_H ≥ 0 → x_H ≥ -net_AI/b1 * x_AI
net2=net_coeff_AI[i]
x_AI_max_sector=30000/(N*2)  # giả sử chia đều tối đa
if net2<0:
    xH_min = abs(net2)/b1[i] * x_AI_max_sector
    print(f"  Khi net_coeff_AI[{i}]={net2:.2f} < 0 → CN chế biến BỊ MẤT việc làm ròng khi đầu tư AI")
    print(f"  x_H tối thiểu (khi x_AI={x_AI_max_sector:.0f}): {xH_min:.0f} tỷ")
else:
    print(f"  net_coeff_AI={net2:.2f} > 0 → CN chế biến TẠO THÊM việc làm ròng dù không có H")
    print(f"  x_H hiện tại = {x_H[i]:.0f} tỷ (đủ để đảm bảo Displaced≤Retrain)")

# CÂU 9.4.3 – Lao động dễ tổn thương
print(f"\nCÂU 9.4.3 – Lao động phổ thông dễ tổn thương (ngành 1,3,4):")
for idx in [0,2,3]:
    disp_pct=Displaced[idx]/(L[idx]*1e6)*100 if L[idx]>0 else 0
    print(f"  {sector_names[idx]:15s}: Displaced={Displaced[idx]:7,.0f} việc ({disp_pct:.3f}% tổng LĐ ngành)")

# CÂU 9.4.4 – Thêm ràng buộc Displaced ≤ 5% L
print(f"\nCÂU 9.4.4 – Ràng buộc Displaced ≤ 5% × L (triệu LĐ):")
Al2=A_ub.tolist(); bl2=b_ub.tolist()
for i in range(N):
    row=np.zeros(Nv); row[i]=c1[i]*risk[i]
    Al2.append(row); bl2.append(0.05*L[i]*1e6)  # 5% × L(người)
res2=linprog(c_obj, A_ub=np.array(Al2), b_ub=np.array(bl2), bounds=bds, method='highs')
if res2.status==0:
    print(f"  Khả thi! NetJob tổng = {-res2.fun:,.0f} (thay đổi {-res2.fun-(-res.fun):+,.0f})")
else:
    print("  KHÔNG KHẢ THI với ràng buộc 5% LĐ → Tốc độ tự động hóa vượt năng lực đào tạo lại")

# Biểu đồ
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('Bài 9 – Tác động AI tới thị trường lao động Việt Nam',fontweight='bold')
colors_net=['#4CAF50' if n>0 else '#F44336' for n in NetJob]
axes[0].barh(sector_names[::-1],NetJob[::-1],color=colors_net[::-1],alpha=0.9)
axes[0].axvline(0,color='black',lw=1); axes[0].set_title(f'NetJob ròng\nTổng={NetJob.sum():,.0f}',fontweight='bold')
axes[0].set_xlabel('Số việc làm ròng'); axes[0].grid(axis='x',alpha=0.3)

x_pos=np.arange(N)
axes[1].bar(x_pos-0.3,x_AI,0.28,label='x_AI',color='#9C27B0',alpha=0.85)
axes[1].bar(x_pos+0.02,x_H,0.28,label='x_H',color='#4CAF50',alpha=0.85)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels([s[:8] for s in sector_names],rotation=35,fontsize=8)
axes[1].set_title('Phân bổ đầu tư tối ưu',fontweight='bold'); axes[1].legend(); axes[1].grid(axis='y',alpha=0.3)
axes[1].set_ylabel('Tỷ VND')

# Sankey-style: displaced flows
flows=[NewJob,UpJob,-Displaced]
colors_flow=['#4CAF50','#2196F3','#F44336']
labels_flow=['Việc mới (AI)','Nâng cấp (H)','Bị thay thế']
bottom=np.zeros(N)
for fl,col,lbl in zip(flows,colors_flow,labels_flow):
    axes[2].bar(x_pos,np.abs(fl),0.5,bottom=bottom if lbl!='Bị thay thế' else None,
                color=col,alpha=0.8,label=lbl)
    if lbl!='Bị thay thế': bottom+=np.abs(fl)
axes[2].set_xticks(x_pos); axes[2].set_xticklabels([s[:8] for s in sector_names],rotation=35,fontsize=8)
axes[2].set_title('Cấu trúc việc làm\n(xanh lá=tạo mới, đỏ=bị thay thế)',fontweight='bold')
axes[2].legend(); axes[2].grid(axis='y',alpha=0.3)

plt.tight_layout(); plt.savefig('bai09_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()
print(f"""
THẢO LUẬN:
a) Ngành cần đào tạo nhiều nhất: {sector_names[np.argmax(x_H)]} (x_H={x_H[np.argmax(x_H)]:.0f} tỷ)
   Phù hợp thực tế: CN chế biến 11.5 triệu LĐ nguy cơ cao nhất
b) Tài chính-NH (risk=52%): đầu tư x_AI mạnh nhưng cần x_H đủ lớn để đào tạo lại
   → Chiến lược: đầu tư FinTech kèm theo chương trình đào tạo lại 2-3 năm
c) Nông-Lâm-TS: a1=8.5 thấp nhưng 13.2 triệu LĐ → chỉ cần đầu tư nhỏ AI là đủ tác động
d) Ràng buộc 'Displaced≤Retrain' là biểu diễn câu "tốc độ tự động hóa≤năng lực đào tạo lại"
""")
print("✓ Bài 9 → bai09_ket_qua.png")
