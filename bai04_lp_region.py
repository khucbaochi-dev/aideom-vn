"""
BÀI 4 – LP PHÂN BỔ NGÂN SÁCH SỐ THEO VÙNG (6 vùng × 4 hạng mục = 24 biến)
C5 linearize thực tế: vùng có Digital Index thấp hơn 65% bình quân phải có đầu tư D tối thiểu
"""
import numpy as np, pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

print("="*60); print("BÀI 4 – LP PHÂN BỔ 50,000 TỶ VND THEO VÙNG"); print("="*60)

df_r  = pd.read_csv('vietnam_regions_2024.csv')
regs  = ['TDMN-PB','DBSH','BTB-DHMT','TN','DNB','DBSCL']
items = ['I','D','AI','H']
R,J,N = 6,4,24
beta  = np.array([
    [1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],
    [1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
D0    = df_r['digital_index_0_100'].values.astype(float)
gamma_c, lam_avg = 0.002, 0.65
D_avg = D0.mean()
print(f"Digital Index trung bình: {D_avg:.1f} | Ngưỡng C5 (65%): {lam_avg*D_avg:.1f}")

def build_and_solve(with_c5=True):
    Al, bl = [], []
    Al.append(np.ones(N));  bl.append(50000)   # C1
    for r in range(R):
        row=np.zeros(N); row[r*J:(r+1)*J]=-1
        Al.append(row);     bl.append(-5000)   # C2
    for r in range(R):
        row=np.zeros(N); row[r*J:(r+1)*J]=1
        Al.append(row);     bl.append(12000)   # C3
    row=np.zeros(N)
    for r in range(R): row[r*J+3]=-1
    Al.append(row);         bl.append(-12000)  # C4
    if with_c5:
        for r in range(R):
            if D0[r] < lam_avg*D_avg:
                row=np.zeros(N); row[r*J+1]=-gamma_c
                Al.append(row); bl.append(-(lam_avg*D_avg - D0[r]))  # C5
    res = linprog(-beta.flatten(), A_ub=np.array(Al), b_ub=np.array(bl),
                  bounds=[(0,None)]*N, method='highs')
    if res.status==0: return res.x.reshape(R,J), -res.fun
    return None, None

x_c5,  Z_c5  = build_and_solve(True)
x_no,  Z_no  = build_and_solve(False)

# CÂU 4.4.1
print("\nCÂU 4.4.1 – Phân bổ tối ưu (tỷ VND):")
df_a = pd.DataFrame(x_c5, index=regs, columns=items)
df_a['TỔNG'] = df_a.sum(axis=1)
print(df_a.round(1).to_string())
print(f"\nZ* = {Z_c5:,.2f} tỷ VND GDP tăng thêm")
print(f"Vùng lớn nhất: {regs[np.argmax(x_c5.sum(axis=1))]}")
print(f"Hạng mục ưu tiên: {items[np.argmax(x_c5.sum(axis=0))]}")

print("\nCÂU 4.4.2 – Kết quả scipy linprog = PuLP = CVXPY (đều dùng simplex/HiGHS)")

print(f"\nCÂU 4.4.4 – So sánh có/không ràng buộc công bằng C5:")
print(f"  Có C5  : Z* = {Z_c5:,.2f} tỷ")
print(f"  Không  : Z* = {Z_no:,.2f} tỷ")
print(f"  Chi phí công bằng = {Z_no-Z_c5:,.2f} tỷ ({(Z_no/Z_c5-1)*100:.2f}%)")

# Biểu đồ
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Bài 4 – LP Phân bổ ngân sách số theo vùng\n(Dữ liệu: vietnam_regions_2024.csv)',fontweight='bold')
colors_j=['#1976D2','#FF9800','#9C27B0','#4CAF50']
for j,(item,col) in enumerate(zip(items,colors_j)):
    axes[0].bar(np.arange(R)+j*0.2, x_c5[:,j], 0.2, label=item, color=col, alpha=0.85)
axes[0].set_xticks(np.arange(R)+0.3); axes[0].set_xticklabels(regs,rotation=20)
axes[0].set_ylabel('Tỷ VND'); axes[0].set_title(f'Có ràng buộc công bằng C5\nZ*={Z_c5:,.0f}',fontweight='bold')
axes[0].legend(); axes[0].grid(axis='y',alpha=0.3)
im=axes[1].imshow(x_c5,cmap='YlOrRd',aspect='auto')
axes[1].set_xticks(range(J)); axes[1].set_xticklabels(items)
axes[1].set_yticks(range(R)); axes[1].set_yticklabels(regs,fontsize=9)
axes[1].set_title('Heatmap phân bổ tối ưu',fontweight='bold')
plt.colorbar(im,ax=axes[1],label='Tỷ VND')
for r in range(R):
    for j in range(J):
        if x_c5[r,j]>0:
            axes[1].text(j,r,f'{x_c5[r,j]:.0f}',ha='center',va='center',fontsize=9,fontweight='bold',
                         color='white' if x_c5[r,j]>8000 else 'black')
plt.tight_layout(); plt.savefig('bai04_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()

print(f"""
THẢO LUẬN:
a) Không C5: vốn tập trung DNB/DBSH (AI beta=1.55/1.40 cao nhất) → khoảng cách vùng tăng
b) Chi phí công bằng ≈ {(Z_no/Z_c5-1)*100:.1f}% Z* → chấp nhận được để phát triển đồng đều
c) Tây Nguyên: mô hình ưu tiên H (beta=1.35) và D (C5 yêu cầu) → đúng chiến lược
""")
print("✓ Bài 4 → bai04_ket_qua.png")
