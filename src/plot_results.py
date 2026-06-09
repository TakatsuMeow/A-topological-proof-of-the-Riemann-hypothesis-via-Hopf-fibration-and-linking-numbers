import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Загружаем результаты из Rust
with open('results/rust_results.json', 'r') as f:
    data = json.load(f)

sigmas = np.array(data['sigma_values'])
eff = np.array(data['eff_dims'])
ldet = np.array(data['log_dets'])
corr_z = np.array(data['corr_zeta'])
corr_x = np.array(data['corr_xi'])
conds = np.array(data['conds'])
shifts = np.array(data['stirling_shifts'])
t_sample = np.array(data['t_sample'])

# Стирлинг: подбираем модель
models = {'A/t': lambda t, A: A/t, 'A/ln(t)': lambda t, A: A/np.log(t)}
best_m, best_r2, best_p = None, -np.inf, None
for name, fn in models.items():
    try:
        popt, _ = curve_fit(lambda t, A: fn(t, A), t_sample, shifts)
        pred = fn(t_sample, *popt)
        r2 = 1 - np.sum((shifts-pred)**2)/np.sum((shifts-np.mean(shifts))**2)
        if r2 > best_r2: best_m, best_r2, best_p = name, r2, popt
    except: pass

# Z-scores
i05 = np.where(sigmas == 0.5)[0][0]
others = [i for i in range(len(sigmas)) if i != i05]

all_zs = []
for vals in [eff, ldet, corr_z, corr_x]:
    v05, v_oth = vals[i05], np.array([vals[i] for i in others])
    z = abs((v05-np.mean(v_oth))/np.std(v_oth)) if np.std(v_oth) > 0 else 0
    all_zs.append(z)

max_z = max(all_zs)

# Графики
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

ax = axes[0,0]; ax.plot(sigmas, eff, 'b-o', lw=2, ms=10); ax.axvline(0.5, color='red', ls='--', lw=2)
ax.set_xlabel('σ'); ax.set_ylabel('Эфф. размерность'); ax.set_title('Гильбертово пространство'); ax.grid(True, alpha=0.3)

ax = axes[0,1]; ax.plot(sigmas, ldet, 'g-D', lw=2, ms=10); ax.axvline(0.5, color='red', ls='--', lw=2)
ax.set_xlabel('σ'); ax.set_ylabel('log pseudodet'); ax.set_title('Псевдоопределитель'); ax.grid(True, alpha=0.3)

ax = axes[0,2]; ax.plot(sigmas, corr_z, 'b-o', lw=2, ms=8, label='ζ(s)'); ax.plot(sigmas, corr_x, 'r-s', lw=2, ms=8, label='ξ(s)')
ax.axvline(0.5, color='red', ls='--'); ax.set_xlabel('σ'); ax.set_ylabel('Корреляция')
ax.set_title('ζ(s) vs ξ(s)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,0]; ax.loglog(t_sample, shifts, 'b-', lw=2, alpha=0.7)
if best_m == 'A/t': ax.loglog(t_sample, best_p[0]/t_sample, 'r--', lw=2, label=f'{best_p[0]:.2f}/t')
else: ax.loglog(t_sample, best_p[0]/np.log(t_sample), 'r--', lw=2, label=f'{best_p[0]:.2f}/ln(t)')
ax.set_xlabel('t'); ax.set_ylabel('|Δσ|'); ax.set_title(f'Сдвиг Стирлинга ({best_m})'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,1]; t_ext = np.logspace(1, 20, 100)
drift = best_p[0]/t_ext if best_m == 'A/t' else best_p[0]/np.log(t_ext)
ax.semilogx(t_ext, 0.5-drift, 'b-', lw=2); ax.axhline(0.5, color='red', ls='--')
ax.fill_between(t_ext, 0.49, 0.51, alpha=0.1, color='red')
ax.set_xlabel('t → ∞'); ax.set_ylabel('Пик σ'); ax.set_title('Экстраполяция к ∞'); ax.grid(True, alpha=0.3)

ax = axes[1,2]; ax.axis('off')
summary = f"ИТОГИ (Rust)\n{'='*20}\n\nN={data['n_zeros']:,}\nВремя: {data['total_time_secs']/60:.0f}м\n\n"
for name, z in [("Эфф.разм", all_zs[0]), ("log det", all_zs[1]), ("Корр ζ", all_zs[2]), ("Корр ξ", all_zs[3])]:
    summary += f"{name}: Z={z:.1f}σ\n"
summary += f"\nДрейф: {best_m}"
if max_z > 3: summary += f"\n\n★★★ {max_z:.0f}σ ★★★"
elif max_z > 2: summary += f"\n\n★★ {max_z:.1f}σ ★★"

ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle(f'СТАТИЧЕСКИЙ ОКЕАН — Rust-ускорение — {data["n_zeros"]:,} нулей', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/final_rust.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nMax Z-score: {max_z:.1f}σ")
print(f"График: results/final_rust.png")