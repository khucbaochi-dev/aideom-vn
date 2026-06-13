"""
BÀI 3 – CHỈ SỐ ƯU TIÊN NGÀNH (Priority Index)
Đọc từ vietnam_sectors_2024.csv thực tế
Công thức: Priority_i = Σ a_j * x̃_j(good) - a_risk * x̃_risk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("BÀI 3 – CHỈ SỐ ƯU TIÊN NGÀNH VIỆT NAM 2024")
print("="*60)

df = pd.read_csv('vietnam_sectors_2024.csv')
# Thêm tên tiếng Việt
vi_names = ['Nông-Lâm-TS','CN Chế biến','Xây dựng','Khai khoáng','Bán buôn-lẻ',
            'Tài chính-NH','Logistics','CNTT-TT','Giáo dục','Y tế']
df['sector_vi'] = vi_names

# Thêm cột productivity (từ đề bài vì CSV dùng gdp_share thay thế)
productivity = np.array([103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1])
df['productivity_mil_VND'] = productivity

print("\nDữ liệu từ CSV:")
print(df[['sector_vi','growth_rate_2024_pct','productivity_mil_VND','spillover_coef_0_1',
          'export_billion_USD','labor_million','ai_readiness_0_100','automation_risk_pct']].to_string(index=False))

# ── CÂU 3.4.1: CHUẨN HÓA MIN-MAX ─────────────────────────────────────────
def norm_good(s): return (s - s.min()) / (s.max() - s.min() + 1e-10)
def norm_bad(s):  return (s.max() - s) / (s.max() - s.min() + 1e-10)

cols_good = ['growth_rate_2024_pct','productivity_mil_VND','spillover_coef_0_1',
             'export_billion_USD','labor_million','ai_readiness_0_100']
X_norm = pd.DataFrame(index=df.index)
for c in cols_good:
    X_norm[c] = norm_good(df[c])
X_norm['risk_inv'] = norm_bad(df['automation_risk_pct'])

print("\nCÂU 3.4.1 – Ma trận đã chuẩn hóa (0-1):")
print(X_norm.round(3).to_string())

# ── CÂU 3.4.2: TÍNH PRIORITY VỚI TRỌNG SỐ MẶC ĐỊNH ───────────────────────
w = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
w_risk = 0.15

priority = X_norm[cols_good].values @ w + w_risk * X_norm['risk_inv'].values
df['Priority'] = priority
df_sorted = df.sort_values('Priority', ascending=False)

print("\nCÂU 3.4.2 – Xếp hạng Priority:")
print("-"*45)
for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
    bar = '█' * int(row['Priority']*30)
    print(f"  #{rank:2d} {row['sector_vi']:15s} {row['Priority']:.4f}  {bar}")

# ── CÂU 3.4.3: ĐỘ NHẠY VỚI a6 ───────────────────────────────────────────
a6_range = np.arange(0.05, 0.45, 0.05)
rank_matrix = np.zeros((10, len(a6_range)))

for j, a6 in enumerate(a6_range):
    w_temp = np.array([0.15, 0.15, 0.20, 0.15, 0.10, a6])
    # chuẩn hóa 6 trọng số "good" về tổng = 1-w_risk = 0.85
    w_temp = w_temp / w_temp.sum() * 0.85
    p = X_norm[cols_good].values @ w_temp + 0.15 * X_norm['risk_inv'].values
    rank_matrix[:, j] = pd.Series(p).rank(ascending=False).values

print("\nCÂU 3.4.3 – Top-3 theo a6:")
print(f"{'a6':>5}  {'Hạng 1':>15}  {'Hạng 2':>15}  {'Hạng 3':>15}")
for j, a6 in enumerate(a6_range):
    top3_idx = np.argsort(rank_matrix[:, j])[:3]
    names = [vi_names[i] for i in top3_idx]
    print(f"  {a6:.2f}  {names[0]:>15}  {names[1]:>15}  {names[2]:>15}")

# ── CÂU 3.4.4: HAI BỘ TRỌNG SỐ ──────────────────────────────────────────
w_gr = np.array([0.30, 0.25, 0.08, 0.25, 0.02, 0.05]); w_gr /= w_gr.sum()*1/0.85
w_in = np.array([0.08, 0.05, 0.22, 0.05, 0.28, 0.10]); w_in /= w_in.sum()*1/0.85
p_gr = X_norm[cols_good].values @ w_gr + 0.15 * X_norm['risk_inv'].values
p_in = X_norm[cols_good].values @ w_in + 0.15 * X_norm['risk_inv'].values

rk_gr = pd.Series(p_gr).rank(ascending=False).astype(int)
rk_in = pd.Series(p_in).rank(ascending=False).astype(int)

print("\nCÂU 3.4.4 – So sánh hai bộ trọng số:")
print(f"  {'Ngành':15s}  {'Tăng trưởng':>12}  {'Bao trùm':>10}  {'ΔHạng':>7}")
for i, name in enumerate(vi_names):
    delta = int(rk_in.iloc[i]) - int(rk_gr.iloc[i])
    sign = '+' if delta > 0 else ''
    print(f"  {name:15s}  {'#'+str(int(rk_gr.iloc[i])):>12}  {'#'+str(int(rk_in.iloc[i])):>10}  {sign+str(delta):>7}")

# ── BIỂU ĐỒ ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Bài 3 – Chỉ số ưu tiên ngành Việt Nam 2024\n(Dữ liệu: vietnam_sectors_2024.csv)',
             fontweight='bold', fontsize=12)

sort_idx = np.argsort(priority)[::-1]
colors_p = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
axes[0].barh([vi_names[i] for i in sort_idx], [priority[i] for i in sort_idx],
             color=[colors_p[k] for k in range(10)], alpha=0.9)
axes[0].axvline(np.median(priority), color='red', ls='--', alpha=0.6, label=f'Median={np.median(priority):.3f}')
axes[0].set_title('Xếp hạng Priority\n(trọng số mặc định)', fontweight='bold')
axes[0].set_xlabel('Điểm Priority'); axes[0].legend(); axes[0].grid(axis='x', alpha=0.3)

im = axes[1].imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=10)
axes[1].set_xticks(range(len(a6_range)))
axes[1].set_xticklabels([f'{a:.2f}' for a in a6_range], rotation=45, fontsize=8)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([n[:10] for n in vi_names], fontsize=8)
axes[1].set_title('Heatmap xếp hạng theo a6\n(xanh=hạng cao, đỏ=thấp)', fontweight='bold')
axes[1].set_xlabel('Trọng số AI Readiness (a6)')
for i in range(10):
    for j in range(len(a6_range)):
        axes[1].text(j, i, int(rank_matrix[i,j]), ha='center', va='center',
                     fontsize=7, color='white' if rank_matrix[i,j] <= 3 or rank_matrix[i,j] >= 8 else 'black')
plt.colorbar(im, ax=axes[1], label='Hạng')

w2 = 0.35
x_pos = np.arange(10)
axes[2].bar(x_pos-w2/2, p_gr, w2, label='Tăng trưởng', color='#1976D2', alpha=0.85)
axes[2].bar(x_pos+w2/2, p_in, w2, label='Bao trùm',   color='#4CAF50', alpha=0.85)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([n[:8] for n in vi_names], rotation=40, ha='right', fontsize=8)
axes[2].set_title('So sánh trọng số\nTăng trưởng vs Bao trùm', fontweight='bold')
axes[2].set_ylabel('Priority score'); axes[2].legend(); axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('bai03_ket_qua.png', dpi=150, bbox_inches='tight')
plt.close()

top3_vi  = df_sorted['sector_vi'].iloc[:3].tolist()
print("\n" + "="*60)
print("THẢO LUẬN CHÍNH SÁCH")
print("="*60)
print(f"""
a) Top-3 ưu tiên chuyển đổi số & AI:
   1. {top3_vi[0]}  2. {top3_vi[1]}  3. {top3_vi[2]}
   → Nghị quyết 57-NQ/TW xác định CNTT, CN chế tạo, tài chính số là trọng tâm → PHÙ HỢP

b) Khai khoáng có năng suất 1.290 triệu VND/LĐ nhưng không vào top vì:
   - Tốc độ tăng trưởng ÂM (-1.2%) → điểm growth = 0
   - Rủi ro tự động hóa 55% (cao nhất) → bị trừ điểm nặng
   - AI Readiness chỉ 30/100 → hệ số lan tỏa thấp (0.30)

c) Quyết định trọng số – ai quyết?
   - Chuyên gia kỹ thuật: xây dựng phương pháp và đo lường
   - Hội đồng chính sách: phê duyệt ưu tiên chiến lược quốc gia
   - Tham vấn công khai (public consultation): bảo đảm chính danh dân chủ
   → Mô hình AIDEOM-VN nên là công cụ hỗ trợ, không tự quyết định trọng số
""")
print("✓ Bài 3 hoàn thành → bai03_ket_qua.png")
