# A-B-tests-pet-project
В данной работе представлен пет проект по аналитике данных. Генерируются две выборки со смещенным средним и проводятся A/B и A/A тесты. Язык - Python

Задача:
Оценка модели прогнозирования CTR (кликабельности) для рекламы. Контролируемые эксперименты (A/B-тесты) — это самый надежный способ в данном случае.

Данные:
Данные генерируются с помощью функции generate_data. Предположим, что то, сколько раз пользователь увидел рекламу (views) распределено логнормально (большинство пользователей увидело мало рекламы, но некоторые - много). Пусть CTR пользователя (user CTR) описывается бета распределением (хорошо подходит, потому что область определения - [0, 1]). Тогда количество переходов на рекламный сайт (clicks) распределено биномиально, где число испытаний - это views, а вероятность успеха p - это user CTR.
<details>
   <summary>Генерация данных
      
   ```python
   def generate_data(n_users, success_rate, beta, uplift=0):
     mean = success_rate * (1 + uplift)
     alpha = (mean * beta) / (1 - mean)
    
     user_ctrs = np.random.beta(alpha, beta, size=n_users)
     views = np.random.lognormal(mean=1, sigma=2.0, size=n_users).astype(int) + 1
     clicks = np.random.binomial(n=views, p=user_ctrs)
    
     return views, clicks, user_ctrs
   ```
   </details>
1. Задаем N (количество пользователей) для контрольной и тестовой групп. В данном случай N = 5000.
2. Генерируем количество увиденной рекламы одним пользователем (views) для тестовой и контрольной групп, используя одно и то же распределение
3. Генерируем user CTR для контрольной группы, где среднее равно success_rate, а для тестовой группы - success_rate * (1 + uplift). Пусть uplift = 0.2.
4. Генерируем clicks для обеих групп.
![Сгенерированные данные](slides/generated_data.png)
Проделав эти шаги один раз, мы получаем один A/B тест. Проделав это NN раз мы получим NN A/B тестов (метод Монте-Карло). Положим NN равным 2000.

A/B и A/A тесты:
В основном нулевая гипотеза звучит так: H_0 - средние контрольной и тестовой групп равны, а H_1 - средние не равны. Уровень значимости примем равным 0.05.
Для каждого A/B-теста мы проведём ряд статистических тестов и сравним их эффективность, построив ROC-кривую. Также расчитаем мощность каждого теста
Используемые тесты:
1. тест Манна-Уитни
2. t-тест
3. Пуассоновский бутстрэп
4. Бакетизация
Для каждого статистического теста мы должны иметь возможность проверить, контролирует ли p-значение FPR для наших данных. Чтобы это сделать, нам нужно сгенерировать A/A-тесты. Используем те же шаги генерации данных, но с uplift = 0. Значит распределения тестовой и контрольной групп будут одинаковы, следовательно, нулевая гипотеза (H₀) здесь будет истинна. p-value корректно контролирует FPR тогда и только тогда, когда для A/A-тестов p-значение распределено равномерно на отрезке [0, 1], а ROC-кривая имеет диагональный вид. Также для для оценки консервативности и антиконсервативности теста будем рассчитывать доверительный интервал Уилсона.

Тесты и результаты:
1. Тест Манна Уитни
   Используем функцию stats.mannwhitneyu, которая будет сравнивать медианы количеств нажатий на рекламу для групп A и B (clicks).
   ![Сгенерированные данные](slides/mannwhitneyu.png)
   <details>
   <summary>A/B тест. Манн-Уитни
      
   ```python
    #A/B тест (тест Манна-Уитни)

    NN = 2000
    pvalues = []
    cnt = 0
    #Monte-Carlo
    for _ in tqdm(range(NN)):
        views_a, clicks_a, user_ctrs_a = generate_data(N, success_rate, beta)
        views_b, clicks_b, user_ctrs_b = generate_data(N, success_rate, beta, uplift=uplift)
        reg = stats.mannwhitneyu(clicks_a, clicks_b)
        cnt += (reg.pvalue < alpha)
        pvalues.append(reg.pvalue)

    sns.histplot(x = pvalues)
    plt.xlabel('p values', fontsize = 15)
    plt.xlim(-0.05, 1)
    plt.ylim(0, 1300)
    plt.ylabel('Counts', fontsize = 15)
    plt.savefig('pvalues_mannwhitneyu.png', dpi=300, bbox_inches='tight') 
    plt.show()

    df_plot_ROC = pd.DataFrame({
        'FPR': np.sort(pvalues),
        'Sensitivity': [i / NN for i in range(NN)]
    })
    sns.lineplot(data=df_plot_ROC, x='FPR', y='Sensitivity', linewidth=4, color='steelblue')
    plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='-', alpha=0.5)
    plt.axvline(x=alpha, color='grey', linestyle='-', linewidth=2, alpha=0.7)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('Sensitivity', fontsize=15)
    plt.title(f'Power = {int(np.mean(np.array(pvalues) < alpha) * 100)}%', fontsize=15, pad=20)
    plt.savefig('ROC_mannwhitneyu.png', dpi=300, bbox_inches='tight')  
    plt.show()
   ```
   </details>
   <details>
   <summary>A/A тест. Манн-Уитни
      
   ```python
    #A/A тест (тест Манна-Уитни)

    NN = 2000
    pvalues = []
    cnt = 0
    #Monte-Carlo
    for _ in tqdm(range(NN)):
        views_a, clicks_a, user_ctrs_a = generate_data(N, success_rate, beta)
        views_b, clicks_b, user_ctrs_b = generate_data(N, success_rate, beta)
        reg = stats.mannwhitneyu(clicks_a, clicks_b)
        cnt += (reg.pvalue < alpha)
        pvalues.append(reg.pvalue)
    print(np.mean(np.array(pvalues) < alpha))
    print(proportion.proportion_confint(count = cnt, nobs = NN, alpha=0.05, method='wilson'))
    
    sns.histplot(x = pvalues)
    plt.xlabel('p values', fontsize = 15)
    plt.xlim(-0.05, 1)
    plt.ylim(0, 400)
    plt.ylabel('Counts', fontsize = 15)
    plt.savefig('pvalues_mannwhitneyu_AA.png', dpi=300, bbox_inches='tight') 
    plt.show()
    
    df_plot_ROC = pd.DataFrame({
            'FPR': np.sort(pvalues),
            'Sensitivity': [i / NN for i in range(NN)]
        })
    sns.lineplot(data=df_plot_ROC, x='FPR', y='Sensitivity', linewidth=4, color='steelblue')
    plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='-', alpha=0.5)
    plt.axvline(x=alpha, color='grey', linestyle='-', linewidth=2, alpha=0.7)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('Sensitivity', fontsize=15)
    plt.title(f'Power = {int(np.mean(np.array(pvalues) < alpha) * 100)}%', fontsize=15, pad=20)
    plt.savefig('ROC_mannwhitneyu_AA.png', dpi=300, bbox_inches='tight')  
    plt.show()
   ```
   </details>
3. Функция для теста Хи-квадрат с построением тепловой карты. Считаем уровень значимости равным 0.05, значит если p_value < 0.05, то мы отвергаем гипотезу $H_0$ и принимаем альтернативную - $H_1$
   <details>
   <summary>Тест Хи-квадрат + тепловая карта</summary>
      
   ```python
   def chi2_test(column, target_column, df):
       contingency_table = pd.crosstab(df[column], df[target_column])
       chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
   
       print(f"Chi-square p-value: {p_val}")
       if p_val < 0.05:
           print("Есть зависимость")
       else:
           print("Переменные независимы")
   
       plt.figure(figsize=(8, 5))
       sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
       plt.title(f'Тепловая карта: {column} vs {target_column}')
       plt.ylabel(f'{column}')
       plt.savefig(f'heat_map_{column}.png')
       plt.show()
   ```
   </details>
4. Рассчет коэффициента V Крамера
   <details>
   <summary>Рассчет коэффициента V Крамера</summary>
      
   ```python
   def cramers_v(column, target_column, dataframe):
       # 1. Строим таблицу сопряженности
       conf_matrix = pd.crosstab(dataframe[column], dataframe[target_column])
       
       # 2. Считаем Хи-квадрат
       chi2 = chi2_contingency(conf_matrix)[0]
       
       # 3. Параметры расчета
       n = conf_matrix.sum().sum() # Общее кол-во наблюдений
       phi2 = chi2 / n
       r, k = conf_matrix.shape
       
       # Формула V Крамера
       # Результат всегда от 0 (нет связи) до 1 (полная зависимость)
       result = np.sqrt(phi2 / min(k - 1, r - 1))
       return result
   ```
   </details>
