"""
BÀI 2 – QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH SỐ
max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4
Giải bằng scipy.optimize.linprog (HiGHS solver)
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

print("="*60)
print("BÀI 2 – LP PHÂN BỔ NGÂN SÁCH CHUYỂN ĐỔI SỐ")
print("="*60)

c      = np.array([-0.85, -1.20, -0.95, -1.35])   # min(-Z)
A_ub   = np.array([
    [ 1,  1,  1,  1],        # tổng ≤ 100
    [-1,  0,  0,  0],        # x1 ≥ 25
    [ 0, -1,  0,  0],        # x2 ≥ 15
    [ 0,  0, -1,  0],        # x3 ≥ 20
    [ 0,  0,  0, -1],        # x4 ≥ 10
    [ 0.35, -0.65,  0.35, -0.65],   # x2+x4 ≥ 35%
])
b_ub   = np.array([100, -25, -15, -20, -10, 0])
bounds = [(0, None)]*4

# ── CÂU 2.4.1 ──────────────────────────────────────────────────────────────
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
x   = res.x;   Z = -res.fun
labels = ['x1-Hạ tầng số','x2-AI & DL','x3-Nhân lực','x4-R&D']

print("\nCÂU 2.4.1 – Phân bổ tối ưu (B=100 nghìn tỷ VND):")
for lb, xi in zip(labels, x):
    print(f"  {lb:20s} = {xi:8.2f} nghìn tỷ VND  ({xi:.1f}%)")
print(f"\nZ* = {Z:.4f} nghìn tỷ VND GDP tăng thêm")
print(f"Ngân sách dùng: {x.sum():.1f}/100")
tech = (x[1]+x[3])/x.sum()*100
print(f"Tỷ trọng CN chiến lược (AI+R&D): {tech:.1f}% (yêu cầu ≥35%)")

# ── CÂU 2.4.2: SHADOW PRICE bằng perturbation ─────────────────────────────
print("\nCÂU 2.4.2 – Shadow price (perturbation eps=1):")
cnames = ['Ngân sách tổng','Sàn hạ tầng (≥25)','Sàn AI (≥15)',
          'Sàn nhân lực (≥20)','Sàn R&D (≥10)','CN chiến lược (≥35%)']
sps = []
for i in range(len(b_ub)):
    b2 = b_ub.copy(); b2[i] += 1
    r2 = linprog(c, A_ub=A_ub, b_ub=b2, bounds=bounds, method='highs')
    sp = (-r2.fun - Z) if r2.success else 0
    sps.append(sp)
    print(f"  {cnames[i]:30s}: {sp:+.4f} nghìn tỷ GDP / nghìn tỷ ngân sách tăng")
print(f"\n→ Mỗi 1 nghìn tỷ tăng ngân sách → GDP tăng thêm {abs(sps[0]):.4f} nghìn tỷ (ROI ≈ {abs(sps[0])*100:.1f}%)")

# ── CÂU 2.4.3: PHÂN TÍCH ĐỘ NHẠY ─────────────────────────────────────────
print("\nCÂU 2.4.3 – Phân tích độ nhạy ngân sách:")
budgets = np.arange(70, 180, 5); Zs = []
for B in budgets:
    b2 = b_ub.copy(); b2[0] = B
    r2 = linprog(c, A_ub=A_ub, b_ub=b2, bounds=bounds, method='highs')
    Zs.append(-r2.fun if r2.success else np.nan)

# ── CÂU 2.4.4: x3 ≥ 30 ────────────────────────────────────────────────────
b3 = b_ub.copy(); b3[3] = -30
r3 = linprog(c, A_ub=A_ub, b_ub=b3, bounds=bounds, method='highs')
print(f"\nCÂU 2.4.4 – Kịch bản ưu tiên nhân lực (x3 ≥ 30):")
if r3.success:
    Z3 = -r3.fun
    for lb, xi in zip(labels, r3.x):
        print(f"  {lb:20s} = {xi:.2f}")
    print(f"Z* = {Z3:.4f}  |  Thay đổi: {Z3-Z:+.4f} ({(Z3/Z-1)*100:+.1f}%)")
    print("→ BÀI TOÁN VẪN KHẢ THI, Z* giảm nhẹ (nhân lực có hệ số 0.95 < R&D 1.35)")

# ── BIỂU ĐỒ ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Bài 2 – LP Phân bổ ngân sách số (scipy linprog)', fontweight='bold')

colors_bar = ['#1976D2','#FF9800','#4CAF50','#9C27B0']
bars = axes[0].bar(range(4), x, color=colors_bar, alpha=0.9, edgecolor='white', width=0.6)
for bar, val in zip(bars, x):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}', ha='center', fontweight='bold')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(['Hạ tầng','AI & DL','Nhân lực','R&D'], fontsize=9)
axes[0].set_ylabel('nghìn tỷ VND'); axes[0].set_title(f'Phân bổ tối ưu\nZ* = {Z:.2f} nghìn tỷ', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

valid = ~np.isnan(Zs)
axes[1].plot(budgets[valid], np.array(Zs)[valid], 'o-', color='#1976D2', lw=2.5, ms=6)
axes[1].axvline(100, color='red', ls='--', alpha=0.7, label='Cơ sở B=100')
axes[1].set_title('Đường cong Z*(B)\nGDP tăng theo ngân sách', fontweight='bold')
axes[1].set_xlabel('Ngân sách (nghìn tỷ VND)'); axes[1].set_ylabel('Z* (nghìn tỷ VND)')
axes[1].legend(); axes[1].grid(alpha=0.3)

shadow_vals = [abs(s) for s in sps]
axes[2].barh(cnames, shadow_vals, color='#E91E63', alpha=0.8)
axes[2].set_title('Shadow Price\n(nghìn tỷ GDP / nghìn tỷ nới lỏng ràng buộc)', fontweight='bold')
axes[2].set_xlabel('Shadow price'); axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('bai02_ket_qua.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("THẢO LUẬN CHÍNH SÁCH")
print("="*60)
print(f"""
a) Shadow price ngân sách: {abs(sps[0]):.4f} → mỗi nghìn tỷ tăng thêm tạo ~{abs(sps[0])*1000:.0f} tỷ GDP mới
   Đây là ROI của vốn công khá cao, phản ánh giai đoạn VN còn nhiều 'low-hanging fruit'

b) R&D hệ số 1.35 cao nhất nhưng sàn chỉ 10:
   - R&D cần thời gian dài để phát huy → rủi ro giải ngân chậm
   - Phải có hạ tầng và nhân lực trước thì R&D mới hiệu quả
   - Đề bài dùng sàn thấp để mô hình 'tự chọn' tối ưu → R&D được chọn tối đa

c) Tỷ lệ 35% CN chiến lược (AI+R&D): kết quả = {tech:.1f}% → ĐẠT
   Thực tế 2025: ngân sách nhà nước ưu tiên hạ tầng giao thông và an sinh ~60%
   → Cần PPP/FDI để bù phần AI+R&D; chính sách thuế ưu đãi cho DN tech
""")
print("✓ Bài 2 hoàn thành → bai02_ket_qua.png")
