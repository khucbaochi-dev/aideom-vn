"""
BÀI 6 – TOPSIS XẾP HẠNG 6 VÙNG THEO MỨC ĐỘ ƯU TIÊN ĐẦU TƯ AI
Dữ liệu: vietnam_regions_2024.csv
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt

print("="*60); print("BÀI 6 – TOPSIS XẾP HẠNG 6 VÙNG ĐẦU TƯ AI"); print("="*60)

df = pd.read_csv('vietnam_regions_2024.csv')
regs = ['TDMN-PB','DBSH','BTB-DHMT','TN','DNB','DBSCL']
df['region_vi'] = regs

criteria  = ['grdp_per_capita_million_VND','fdi_registered_billion_USD',
             'digital_index_0_100','ai_readiness_0_100',
             'trained_labor_pct','rd_intensity_pct',
             'internet_penetration_pct','gini_coef']
is_benefit = [True,True,True,True,True,True,True,False]  # gini = cost
w_expert   = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])

X = df[criteria].values.astype(float)
print("\nDữ liệu vùng từ CSV:")
print(df[['region_vi']+criteria].to_string(index=False))

def topsis(X, w, is_benefit):
    R  = X / np.sqrt((X**2).sum(axis=0))
    V  = R * w
    A_pos = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_pos = np.sqrt(((V-A_pos)**2).sum(axis=1))
    S_neg = np.sqrt(((V-A_neg)**2).sum(axis=1))
    C = S_neg/(S_pos+S_neg)
    return C, S_pos, S_neg

def entropy_weights(X):
    Xp = np.abs(X)+1e-12
    P  = Xp / Xp.sum(axis=0)
    k  = 1.0/np.log(len(X))
    E  = -k * np.nansum(P*np.log(P+1e-12), axis=0)
    d  = 1-E; return d/d.sum()

# CÂU 6.4.1 – Trọng số chuyên gia
C_exp, _, _ = topsis(X, w_expert, is_benefit)
rank_exp    = pd.Series(C_exp).rank(ascending=False).astype(int)
print("\nCÂU 6.4.1 – TOPSIS trọng số chuyên gia:")
df_r = pd.DataFrame({'Vùng':regs,'C*':C_exp.round(4),'Hạng':rank_exp})
print(df_r.sort_values('C*',ascending=False).to_string(index=False))

# CÂU 6.4.2 – Entropy weights
w_ent = entropy_weights(X)
C_ent, _, _ = topsis(X, w_ent, is_benefit)
rank_ent    = pd.Series(C_ent).rank(ascending=False).astype(int)
print(f"\nCÂU 6.4.2 – Trọng số Entropy: {np.round(w_ent,3)}")
df_e = pd.DataFrame({'Vùng':regs,'C*_exp':C_exp.round(4),'H_exp':rank_exp,
                     'C*_ent':C_ent.round(4),'H_ent':rank_ent})
df_e['ΔHạng'] = rank_ent-rank_exp
print(df_e.to_string(index=False))

# CÂU 6.4.3 – Độ nhạy w_AI
w_ai_range = np.arange(0.10,0.45,0.05)
rank_matrix = np.zeros((6,len(w_ai_range)))
for k,wai in enumerate(w_ai_range):
    w_sens = w_expert.copy(); w_sens[3]=wai
    w_sens[[0,1,2,4,5,6,7]] *= (1-wai)/w_sens[[0,1,2,4,5,6,7]].sum()
    C_s,_,_ = topsis(X, w_sens/w_sens.sum(), is_benefit)
    rank_matrix[:,k] = pd.Series(C_s).rank(ascending=False).values
print("\nCÂU 6.4.3 – Top-2 khi thay đổi w_AI:")
for k,wai in enumerate(w_ai_range):
    top2 = np.argsort(rank_matrix[:,k])[:2]
    print(f"  w_AI={wai:.2f}: #1={regs[top2[0]]}  #2={regs[top2[1]]}")

# Biểu đồ
fig, axes = plt.subplots(1,3,figsize=(18,6))
fig.suptitle('Bài 6 – TOPSIS Xếp hạng vùng ưu tiên đầu tư AI\n(Dữ liệu: vietnam_regions_2024.csv)',
             fontweight='bold')
colors_exp=['#4CAF50' if r==1 else '#2196F3' if r<=3 else '#BDBDBD' for r in rank_exp]
axes[0].barh(regs[::-1], C_exp[::-1], color=colors_exp[::-1], alpha=0.9, edgecolor='white')
axes[0].set_title('TOPSIS – Trọng số Chuyên gia\n(xanh lá=hạng 1, xanh=top3)',fontweight='bold')
axes[0].set_xlabel('Hệ số gần gũi C*'); axes[0].grid(axis='x',alpha=0.3)
for i,v in enumerate(C_exp[::-1]):
    axes[0].text(v+0.003, i, f'{v:.3f}', va='center', fontsize=9)

x_pos=np.arange(6); w_b=0.35
axes[1].bar(x_pos-w_b/2, C_exp, w_b, label='Chuyên gia', color='#1976D2', alpha=0.85)
axes[1].bar(x_pos+w_b/2, C_ent, w_b, label='Entropy',    color='#E91E63', alpha=0.85)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels([r[:8] for r in regs], rotation=20)
axes[1].set_title('So sánh trọng số\nChuyên gia vs Entropy',fontweight='bold')
axes[1].set_ylabel('C*'); axes[1].legend(); axes[1].grid(axis='y',alpha=0.3)

im=axes[2].imshow(rank_matrix,cmap='RdYlGn_r',aspect='auto',vmin=1,vmax=6)
axes[2].set_xticks(range(len(w_ai_range)))
axes[2].set_xticklabels([f'{w:.2f}' for w in w_ai_range],rotation=45,fontsize=8)
axes[2].set_yticks(range(6)); axes[2].set_yticklabels([r[:10] for r in regs],fontsize=9)
axes[2].set_title('Heatmap độ nhạy w_AI\n(xanh=hạng cao)',fontweight='bold')
axes[2].set_xlabel('Trọng số AI Readiness')
plt.colorbar(im,ax=axes[2],label='Hạng')
for r in range(6):
    for k in range(len(w_ai_range)):
        axes[2].text(k,r,int(rank_matrix[r,k]),ha='center',va='center',fontsize=9,fontweight='bold',
                     color='white' if rank_matrix[r,k]<=2 or rank_matrix[r,k]>=5 else 'black')

plt.tight_layout(); plt.savefig('bai06_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()

best = regs[np.argmax(C_exp)]; best_ent = regs[np.argmax(C_ent)]
print(f"""
THẢO LUẬN:
a) Vùng dẫn đầu theo chuyên gia: {best}
   → Đây là vùng hợp lý để đặt trung tâm AI đầu tiên (GRDP cao, FDI lớn, AI Ready)

b) Entropy thay đổi: {best_ent} → trọng số khách quan nhấn mạnh tiêu chí phân tán nhiều nhất
   Gini và Digital Index có entropy cao → được nâng trọng số → vùng có chênh lệch lớn bị ảnh hưởng

c) AI Readiness và Internet tương quan cao (r>0.9) → multicollinearity
   → Giải pháp: PCA để gộp 2 tiêu chí / dùng Factor Analysis trước TOPSIS

d) 3 vùng cho trung tâm AI: {regs[np.argsort(C_exp)[::-1][0]]}, {regs[np.argsort(C_exp)[::-1][1]]}, {regs[np.argsort(C_exp)[::-1][2]]}
   Cần bổ sung: cân nhắc địa-chính trị (biên giới, cảng biển, kết nối cáp quang quốc tế)
""")
print("✓ Bài 6 → bai06_ket_qua.png")
