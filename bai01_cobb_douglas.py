"""
BÀI 1 – HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
Dữ liệu: vietnam_macro_2020_2025.csv (thực tế từ thầy)
Y_t = A_t * K^alpha * L^beta * D^gamma * AI^delta * H^theta
alpha+beta+gamma+delta+theta = 1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── ĐỌC DỮ LIỆU THỰC TẾ ────────────────────────────────────────────────────
df = pd.read_csv('vietnam_macro_2020_2025.csv')
print("Dữ liệu macro đọc từ CSV:")
print(df[['year','GDP_trillion_VND','digital_economy_share_GDP_pct',
          'FDI_disbursed_billion_USD','labor_productivity_million_VND']].to_string(index=False))

years = df['year'].values
Y     = df['GDP_trillion_VND'].values   # nghìn tỷ VND

# Các biến cần thêm từ đề bài (không có trong CSV → dùng bảng đề bài)
K  = np.array([16500, 17800, 19600, 21300, 23500, 25900])   # nghìn tỷ VND
L  = np.array([53.6,  50.5,  51.7,  52.4,  52.9,  53.4])   # triệu lao động
D  = df['digital_economy_share_GDP_pct'].values              # % GDP (từ CSV!)
AI = np.array([55.6,  60.2,  65.4,  67.0,  73.8,  80.1])   # nghìn DN số
H  = np.array([24.1,  26.1,  26.2,  27.0,  28.4,  29.2])   # % LĐ qua đào tạo

alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
print(f"\nKiểm tra: alpha+beta+gamma+delta+theta = {alpha+beta+gamma+delta+theta}")

# ══ CÂU 1.4.1: ƯỚC LƯỢNG TFP A_t ══════════════════════════════════════════
A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)

print("\n" + "="*60)
print("CÂU 1.4.1 – Năng suất nhân tố tổng hợp A_t")
print("="*60)
for i, yr in enumerate(years):
    print(f"  {yr}: A = {A[i]:.6f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bài 1 – Hàm sản xuất Cobb-Douglas mở rộng\nDữ liệu Việt Nam 2020-2025',
             fontsize=13, fontweight='bold')

axes[0,0].plot(years, A, 'o-', color='#1976D2', lw=2.5, ms=9)
axes[0,0].fill_between(years, A, alpha=0.15, color='#1976D2')
for i, (yr, a) in enumerate(zip(years, A)):
    axes[0,0].annotate(f'{a:.4f}', (yr, a), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8)
axes[0,0].set_title('TFP (A_t) theo năm', fontweight='bold')
axes[0,0].set_xlabel('Năm'); axes[0,0].set_ylabel('Giá trị TFP'); axes[0,0].grid(alpha=0.3)

# ══ CÂU 1.4.2: DỰ BÁO VÀ MAPE ══════════════════════════════════════════════
A_mean = A.mean()
Y_hat  = A_mean * K**alpha * L**beta * D**gamma * AI**delta * H**theta
mape   = np.mean(np.abs((Y - Y_hat) / Y)) * 100

print("\n" + "="*60)
print(f"CÂU 1.4.2 – Dự báo GDP (A_mean = {A_mean:.6f})")
print("="*60)
df_pred = pd.DataFrame({'Năm': years, 'Y_thực_tế': Y, 'Y_dự_báo': np.round(Y_hat,1),
                        'Sai_số_%': np.round(np.abs((Y-Y_hat)/Y)*100, 2)})
print(df_pred.to_string(index=False))
print(f"\n→ MAPE = {mape:.3f}% → {'XUẤT SẮC' if mape<2 else 'TỐT' if mape<5 else 'KHÁ'}")

axes[0,1].plot(years, Y,     'o-', color='#388E3C', lw=2.5, ms=8, label='Y thực tế')
axes[0,1].plot(years, Y_hat, 's--', color='#D32F2F', lw=2,  ms=7, label=f'Y dự báo (MAPE={mape:.2f}%)')
axes[0,1].set_title('GDP thực tế vs Dự báo', fontweight='bold')
axes[0,1].set_xlabel('Năm'); axes[0,1].set_ylabel('nghìn tỷ VND')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# ══ CÂU 1.4.3: PHÂN RÃ TĂNG TRƯỞNG ═════════════════════════════════════════
dY   = np.diff(np.log(Y))
cK   = alpha * np.diff(np.log(K))
cL   = beta  * np.diff(np.log(L))
cD   = gamma * np.diff(np.log(D))
cAI  = delta * np.diff(np.log(AI))
cH   = theta * np.diff(np.log(H))
cTFP = dY - cK - cL - cD - cAI - cH

periods = [f'{years[i]}-{years[i+1]}' for i in range(len(years)-1)]
components = {'Vốn K': cK, 'Lao động L': cL, 'Số hóa D': cD,
              'Năng lực AI': cAI, 'Nhân lực H': cH, 'TFP': cTFP}

print("\n" + "="*60)
print("CÂU 1.4.3 – Phân rã tăng trưởng (đơn vị: điểm %)")
print("="*60)
print(f"{'Giai đoạn':>12} | {'GDP%':>6} | {'K':>6} | {'L':>7} | {'D':>6} | {'AI':>6} | {'H':>6} | {'TFP':>7}")
print("-"*70)
for i, p in enumerate(periods):
    print(f"{p:>12} | {dY[i]*100:>5.2f}% | {cK[i]*100:>5.2f}% | {cL[i]*100:>6.2f}% | "
          f"{cD[i]*100:>5.2f}% | {cAI[i]*100:>5.2f}% | {cH[i]*100:>5.2f}% | {cTFP[i]*100:>6.2f}%")

means = {k: v.mean()*100 for k,v in components.items()}
print("\nTrung bình đóng góp/năm 2020-2025:")
for k, v in means.items():
    bar = '█' * int(abs(v)*10)
    print(f"  {k:15s}: {v:+6.3f}%  {bar}")

colors_bar = ['#1976D2','#FF9800','#4CAF50','#9C27B0','#F44336','#795548']
x_pos = np.arange(len(periods))
width = 0.13
for i,(name,vals) in enumerate(components.items()):
    axes[1,0].bar(x_pos + i*width, vals*100, width, label=name,
                  color=colors_bar[i], alpha=0.85)
axes[1,0].set_xticks(x_pos + width*2.5)
axes[1,0].set_xticklabels(periods, rotation=20, fontsize=8)
axes[1,0].axhline(0, color='black', lw=0.8)
axes[1,0].set_title('Phân rã đóng góp tăng trưởng GDP', fontweight='bold')
axes[1,0].set_ylabel('Đóng góp (điểm %)'); axes[1,0].legend(fontsize=7); axes[1,0].grid(axis='y', alpha=0.3)

# ══ CÂU 1.4.4: DỰ BÁO 2030 ══════════════════════════════════════════════════
print("\n" + "="*60)
print("CÂU 1.4.4 – Kịch bản dự báo GDP đến 2030")
print("="*60)

A_sim   = A[-1]; K_sim = K[-1]; L_sim = L[-1]
fcst_years = list(range(2025, 2031)); gdp_fc = [Y[-1]]

for yr in range(2026, 2031):
    K_sim  *= 1.06; L_sim  *= 1.06; A_sim  *= 1.012
    t = (yr - 2025)/5
    D_t  = D[-1]  + t * (30.0  - D[-1])
    AI_t = AI[-1] + t * (100.0 - AI[-1])
    H_t  = H[-1]  + t * (35.0  - H[-1])
    Y_t  = A_sim * K_sim**alpha * L_sim**beta * D_t**gamma * AI_t**delta * H_t**theta
    gdp_fc.append(Y_t)

df_fc = pd.DataFrame({'Năm': fcst_years,
                      'GDP_nghìn_tỷ': np.round(gdp_fc,1),
                      'GDP_tỷUSD≈/26': np.round(np.array(gdp_fc)/26,0)})
print(df_fc.to_string(index=False))
cagr_2030 = (gdp_fc[-1]/Y[-1])**(1/5) - 1
print(f"\n→ GDP 2030 ≈ {gdp_fc[-1]:,.0f} nghìn tỷ VND ≈ {gdp_fc[-1]/26:,.0f} tỷ USD")
print(f"→ CAGR 2025-2030: {cagr_2030*100:.2f}%/năm")

# Vẽ forecast
all_y = list(Y) + gdp_fc[1:]
all_yr = list(years) + list(range(2026,2031))
axes[1,1].plot(years, Y, 'o-', color='#388E3C', lw=2.5, ms=8, label='Lịch sử 2020-2025')
axes[1,1].plot(fcst_years, gdp_fc, 's--', color='#E91E63', lw=2.5, ms=8, label='Dự báo 2025-2030')
axes[1,1].axvline(2025, color='gray', linestyle=':', alpha=0.7)
axes[1,1].annotate(f'{gdp_fc[-1]:,.0f}', xy=(2030, gdp_fc[-1]),
                   xytext=(-55,10), textcoords='offset points',
                   fontweight='bold', fontsize=10, color='#E91E63')
axes[1,1].set_title('Dự báo GDP Việt Nam đến 2030\n(D=30%, AI=100k DN, H=35%)', fontweight='bold')
axes[1,1].set_xlabel('Năm'); axes[1,1].set_ylabel('nghìn tỷ VND')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bai01_ket_qua.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("THẢO LUẬN CHÍNH SÁCH")
print("="*60)
print(f"""
a) TFP Việt Nam 2020-2025:
   - Từ {A[0]:.4f} → {A[-1]:.4f} → xu hướng {'TĂNG ổn định' if A[-1]>A[0] else 'GIẢM'}
   - Năm 2021 (COVID) TFP = {A[1]:.4f} thấp nhất → tăng trưởng chủ yếu nhờ vốn K
   - Chất lượng tăng trưởng dần CẢI THIỆN khi TFP phục hồi 2022-2025

b) Đóng góp trung bình D, AI, H:
   - Số hóa D:    {means['Số hóa D']:+.3f}%/năm  ← lớn nhất nhờ tốc độ tăng mạnh 12→19.5%
   - Nhân lực H:  {means['Nhân lực H']:+.3f}%/năm
   - Năng lực AI: {means['Năng lực AI']:+.3f}%/năm

c) Mục tiêu 30% kinh tế số/GDP 2030:
   - Hiện tại (2025): {D[-1]}%, tăng cần thêm {30-D[-1]:.1f}% trong 5 năm
   - Mô hình dự báo khả thi nếu tốc độ ~2.1pp/năm (giai đoạn vừa qua: {(D[-1]-D[0])/5:.1f}pp/năm)
   - Ràng buộc cần: R&D ≥ 3% GDP, đào tạo nhân lực AI 50.000+ kỹ sư
""")
print("✓ Bài 1 hoàn thành → bai01_ket_qua.png")
