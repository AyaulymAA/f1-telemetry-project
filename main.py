import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

fastf1.Cache.enable_cache('f1_cache')

print('Загружаем данные Гран-при Сингапура для анализа ML...')
session = fastf1.get_session(2024, 'Singapore', 'R')
session.load(telemetry=True, laps=True)

ver_lap = session.laps.pick_driver('VER').pick_fastest()
ver_tel = ver_lap.get_car_data().add_distance()

print(f'Точек телеметрии для анализа: {len(ver_tel)}')

print('Запускаем кластеризацию K-Means...')

X = ver_tel[['Distance', 'Speed']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
ver_tel['Cluster'] = kmeans.fit_predict(X)

print('Кластеризация завершена!')

print('Строим график зон торможения...')
plt.figure(figsize=(14, 6))

scatter = plt.scatter(
    ver_tel['Distance'],
    ver_tel['Speed'],
    c=ver_tel['Cluster'],
    cmap='viridis',
    s=5,
    label='Кластеры скорости'
)

plt.xlabel('Дистанция по кругу (метры)', fontsize=12)
plt.ylabel('Скорость (км/ч)', fontsize=12)
plt.title('Автоматический поиск зон торможения (K-Means) — Макс Ферстаппен, Сингапур 2024', fontsize=14)
plt.colorbar(scatter, label='Номер кластера (Зона)')
plt.grid(True, linestyle='--', alpha=0.6)

output_filename = 'verstappen_kmeans_braking_zones.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f'График кластеризации успешно сохранен в файл: {output_filename}')

print('\n--- Анализ: Шарль Леклер ---')
lec_lap = session.laps.pick_driver('LEC').pick_fastest()
lec_tel = lec_lap.get_car_data().add_distance()

X_lec = lec_tel[['Distance', 'Speed']]
kmeans_lec = KMeans(n_clusters=3, random_state=42, n_init=10)
lec_tel['Cluster'] = kmeans_lec.fit_predict(X_lec)

plt.figure(figsize=(14, 6))
scatter_lec = plt.scatter(
    lec_tel['Distance'], lec_tel['Speed'],
    c=lec_tel['Cluster'], cmap='viridis', s=5
)
plt.xlabel('Дистанция по кругу (метры)', fontsize=12)
plt.ylabel('Скорость (км/ч)', fontsize=12)
plt.title('Автоматический поиск зон торможения (K-Means) — Шарль Леклер, Сингапур 2024', fontsize=14)
plt.colorbar(scatter_lec, label='Номер кластера (Зона)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('leclerc_kmeans_braking_zones.png', dpi=300, bbox_inches='tight')
plt.close()
print('График Леклера сохранен!')
