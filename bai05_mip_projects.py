"""
BÀI 5 – MIP LỰA CHỌN DỰ ÁN CHUYỂN ĐỔI SỐ
15 biến nhị phân y_i, giải bằng scipy.optimize.milp (Python 3.9+)
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib.pyplot as plt
import pandas as pd

print("="*60); print("BÀI 5 – MIP LỰA CHỌN DỰ ÁN CHUYỂN ĐỔI SỐ"); print("="*60)

P   = list(range(15))   # 0-based index (P0=P1, P14=P15)
name= ['TTDL Hòa Lạc','TTDL phía Nam','5G toàn quốc','VNeID 2.0','DVCQG v3',
       'Y tế số','Giáo dục K-12','AI quốc gia','Fintech sandbox','Logistics thông minh',
       'Nông nghiệp ĐBSCL','ĐT kỹ sư AI','KCN bán dẫn','An ninh mạng','Open Data']
field=['HT','HT','HT','CP','CP','YT','GD','AI','TC','LG','NN','NL','BD','AN','DL']
C   = [12000,11500,18000,4500,3200,5800,6500,15000,2500,7200,4800,8500,20000,3800,1500]
C1  = [8500,7500,12000,3500,2500,4000,4500,9000,1800,5000,3500,5500,13000,2800,1200]
B   = [21500,20800,32500,9200,6800,11400,12200,28500,5800,13800,8500,16200,35000,7500,3800]
prob= {'HT':0.85,'CP':0.75,'AI':0.65,'BD':0.65,'AN':0.80,'YT':0.80,
       'GD':0.80,'TC':0.80,'LG':0.80,'NN':0.80,'NL':0.80,'DL':0.80}
probs=[prob[f] for f in field]
N   = 15

# ── Hàm giải MIP bằng scipy.optimize.milp ────────────────────────────────
def solve_mip(budget_total=80000, budget_y12=40000,
              force_y1_y2=False, use_expected=False, verbose=True):
    obj = np.array([-B[i]*probs[i] if use_expected else -B[i] for i in P], dtype=float)
    integrality = np.ones(N)      # tất cả binary (milp xử lý 0-1 qua bounds)
    lb = np.zeros(N); ub = np.ones(N)
    if force_y1_y2:               # bắt buộc P1 và P2
        lb[0]=lb[1]=1.0
    bounds = Bounds(lb, ub)

    # Ma trận ràng buộc
    rows, lo_vec, hi_vec = [], [], []
    # C1: Σ C_i*y_i ≤ budget_total
    rows.append(np.array(C,float)); lo_vec.append(-np.inf); hi_vec.append(budget_total)
    # C2: Σ C1_i*y_i ≤ budget_y12
    rows.append(np.array(C1,float)); lo_vec.append(-np.inf); hi_vec.append(budget_y12)
    # C3: y0+y1 ≤ 1
    r=np.zeros(N); r[0]=r[1]=1; rows.append(r); lo_vec.append(-np.inf); hi_vec.append(1)
    # C4: y7 ≤ y11  →  y7-y11 ≤ 0
    r=np.zeros(N); r[7]=1; r[11]=-1; rows.append(r); lo_vec.append(-np.inf); hi_vec.append(0)
    # C5: y12 ≤ y11 → y12-y11 ≤ 0
    r=np.zeros(N); r[12]=1; r[11]=-1; rows.append(r); lo_vec.append(-np.inf); hi_vec.append(0)
    # C6a: y3+y4 ≥ 1 → -(y3+y4) ≤ -1
    r=np.zeros(N); r[3]=1; r[4]=1; rows.append(r); lo_vec.append(1); hi_vec.append(np.inf)
    # C6b: y13 ≥ 1 (bắt buộc)
    r=np.zeros(N); r[13]=1; rows.append(r); lo_vec.append(1); hi_vec.append(np.inf)
    # C7a: Σy ≥ 7
    rows.append(np.ones(N)); lo_vec.append(7); hi_vec.append(11)

    A  = np.array(rows)
    lc = LinearConstraint(A, np.array(lo_vec), np.array(hi_vec))
    res = milp(obj, constraints=lc, integrality=integrality, bounds=bounds)
    if res.status==0:
        chosen = [i for i in P if res.x[i]>0.5]
        Z = -res.fun
        return chosen, Z, res.x
    return [], 0, None

# CÂU 5.4.1 – Cơ sở
chosen, Z, x = solve_mip()
print(f"\nCÂU 5.4.1 – Dự án được chọn (B=80,000 tỷ):")
total_C, total_B = 0, 0
for i in chosen:
    roi = B[i]/C[i]
    print(f"  P{i+1:2d} {name[i]:30s}  Chi phí:{C[i]:6,} tỷ  NPV:{B[i]:6,}  ROI:{roi:.2f}")
    total_C+=C[i]; total_B+=B[i]
print(f"\n  Tổng chi phí: {total_C:,} / 80,000 tỷ  |  Tổng lợi ích Z*= {Z:,.0f} tỷ")
print(f"  NPV biên (Z*/chi phí): {Z/total_C:.3f}")
print(f"  Số dự án chọn: {len(chosen)}")

# CÂU 5.4.2 – Nới B=100,000
chosen2, Z2, _ = solve_mip(100000, 50000)
print(f"\nCÂU 5.4.2 – Nới B=100,000 tỷ: {len(chosen2)} dự án, Z*={Z2:,.0f}")
new_prj = set(chosen2)-set(chosen)
print(f"  Dự án mới được thêm: {[name[i] for i in new_prj]}")

# CÂU 5.4.3 – Bắt buộc cả P1 và P2
chosen3, Z3, _ = solve_mip(force_y1_y2=True)
if chosen3:
    print(f"\nCÂU 5.4.3 – Bắt buộc P1+P2: {len(chosen3)} dự án, Z*={Z3:,.0f} (thay đổi {Z3-Z:+,.0f})")
else:
    print(f"\nCÂU 5.4.3 – Không khả thi khi buộc P1+P2 (vượt ngân sách năm 1-2)")

# CÂU 5.4.4 – Lợi ích kỳ vọng E[Z]
chosen4, EZ4, _ = solve_mip(use_expected=True)
print(f"\nCÂU 5.4.4 – Tối đa E[Z] (có rủi ro): {len(chosen4)} dự án, E[Z*]={EZ4:,.0f}")

# Biểu đồ
fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle('Bài 5 – MIP Lựa chọn dự án chuyển đổi số quốc gia', fontweight='bold')

colors_sel=['#4CAF50' if i in chosen else '#BDBDBD' for i in P]
axes[0].barh(range(N), B, color=colors_sel, alpha=0.85)
axes[0].set_yticks(range(N)); axes[0].set_yticklabels([f'P{i+1}:{name[i][:15]}' for i in P], fontsize=8)
axes[0].axvline(0,color='black',lw=0.5)
axes[0].set_title(f'NPV từng dự án (xanh=được chọn)\nZ*={Z:,.0f} tỷ', fontweight='bold')
axes[0].set_xlabel('NPV (tỷ VND)'); axes[0].grid(axis='x',alpha=0.3)

roi_vals=[B[i]/C[i] for i in P]
colors_roi=['#1976D2' if i in chosen else '#BDBDBD' for i in P]
axes[1].scatter(C, B, c=colors_roi, s=[roi*200 for roi in roi_vals], alpha=0.7, edgecolors='white')
for i in P:
    if i in chosen:
        axes[1].annotate(f'P{i+1}', (C[i],B[i]), fontsize=7, xytext=(3,3), textcoords='offset points')
axes[1].set_title('Chi phí vs NPV (kích thước = ROI)\nXanh=được chọn', fontweight='bold')
axes[1].set_xlabel('Chi phí (tỷ VND)'); axes[1].set_ylabel('NPV (tỷ VND)')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('bai05_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()

# P15 check
p15_roi = B[14]/C[14]
p15_in  = 14 in chosen
print(f"""
THẢO LUẬN:
a) P15 (Open Data): ROI={p15_roi:.2f}, {'ĐƯỢC chọn' if p15_in else 'KHÔNG được chọn'}
   Mặc dù ROI cao nhưng C7 yêu cầu ≤11 dự án → P15 nhường chỗ cho dự án NPV tuyệt đối lớn hơn
   Về chính sách: nên coi P15 là infrastructure → bắt buộc như P14 (an ninh mạng)

b) P14 (An ninh mạng) bắt buộc: làm giảm Z* vì cần chi 3,800 tỷ cho NPV 7,500
   Nhưng AN NINH là điều kiện tiên quyết, không thể lượng hóa hoàn toàn → HỢP LÝ

c) Hiệu ứng cộng hưởng P8×P13: mô hình hóa bằng biến y_synergy = y8*y13,
   linearize: y_syn ≤ y8, y_syn ≤ y13, y_syn ≥ y8+y13-1
   Thêm B_synergy*y_syn vào hàm mục tiêu
""")
print("✓ Bài 5 → bai05_ket_qua.png")
