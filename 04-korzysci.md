## 4. Korzyści z zastosowania Auto-sklearn

Auto-sklearn to narzędzie, które znacząco ułatwia pracę nad modelowaniem uczenia maszynowego, oferując wiele korzyści zarówno dla specjalistów, jak i osób bez specjalistycznej wiedzy technicznej. W tej części przyjrzymy się czterem głównym korzyściom wynikającym z zastosowania Auto-sklearn: redukcji czasu i kosztów, zwiększeniu dostępności modeli dla niefachowców, porównaniu wydajności względem ręcznego modelowania oraz zmniejszeniu ryzyka błędów ludzkich w procesie modelowania.

### 4.1. Redukcja czasu i kosztów budowy modeli

Jedną z największych zalet Auto-sklearn jest **automatyzacja wyboru algorytmów i optymalizacji hiperparametrów**, co prowadzi do ogromnej oszczędności czasu. Budowa modeli uczenia maszynowego tradycyjnie wymagała żmudnych eksperymentów, w których trzeba było ręcznie wybierać i testować różne algorytmy oraz ich hiperparametry. Proces ten mógł trwać wiele dni, tygodni, a nawet miesięcy w zależności od złożoności problemu i danych.

Auto-sklearn eliminuje potrzebę ręcznego eksperymentowania poprzez automatyczne:
- Wybieranie algorytmów,
- Optymalizację hiperparametrów,
- Stosowanie technik ensemblingu,
- Automatyczne przetwarzanie danych.

Dzięki temu firmy mogą szybciej przejść od fazy eksperymentów do fazy produkcji modeli. Zredukowany czas budowy modeli przekłada się również na niższe koszty, ponieważ mniejsza liczba zasobów (takich jak czas pracy specjalistów i infrastruktura obliczeniowa) jest potrzebna do opracowania skutecznych rozwiązań.

#### Przykładowy kod

W poniższym przykładzie pokazano, jak szybko można stworzyć model klasyfikacyjny z użyciem Auto-sklearn w porównaniu do ręcznego modelowania:

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Załadowanie danych
X, y = load_iris(return_X_y=True)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300)

# Trening modelu
automl.fit(X_train, y_train)

# Predykcja i ocena
y_pred = automl.predict(X_test)
print(f'Dokładność: {accuracy_score(y_test, y_pred):.4f}')
```

Ten kod pokazuje, że cały proces budowy modelu, od przetwarzania danych po wybór najlepszego algorytmu i predykcję, odbywa się automatycznie w ciągu kilku minut.

### 4.2. Zwiększenie dostępności zaawansowanych modeli dla niefachowców

Auto-sklearn nie tylko przyspiesza proces budowy modeli, ale także **demokratyzuje dostęp do zaawansowanych technik uczenia maszynowego**. W tradycyjnym podejściu, efektywne budowanie modeli wymagało głębokiej wiedzy z zakresu matematyki, statystyki i programowania. Specjaliści ds. danych (data scientists) musieli znać szczegóły dotyczące algorytmów, technik optymalizacji hiperparametrów, a także różnorodnych metod przetwarzania danych.

Auto-sklearn zmienia ten krajobraz, umożliwiając tworzenie modeli nawet przez osoby o ograniczonej wiedzy technicznej. Ponieważ narzędzie automatyzuje najtrudniejsze aspekty budowy modeli, niefachowcy, tacy jak menedżerowie biznesowi czy analitycy, mogą korzystać z potężnych narzędzi do analizy danych bez potrzeby zdobywania zaawansowanej wiedzy z zakresu uczenia maszynowego.

#### Przykład:

W środowiskach biznesowych, analitycy często muszą tworzyć modele predykcyjne, aby podejmować decyzje dotyczące np. segmentacji klientów lub prognozowania sprzedaży. Dzięki Auto-sklearn mogą skupić się na interpretacji wyników i podejmowaniu działań na podstawie modelu, zamiast na trudnych technicznych aspektach budowy modelu.

### 4.3. Porównanie wydajności względem ręcznego modelowania

Auto-sklearn często osiąga **porównywalne lub lepsze wyniki niż ręczne modelowanie**, szczególnie w przypadkach, gdy testuje się wiele algorytmów. Dzięki technikom takim jak optymalizacja bayesowska, Auto-sklearn skutecznie eksploruje przestrzeń hiperparametrów i wybiera najbardziej optymalne ustawienia. W wielu przypadkach osiąga lepsze wyniki niż ręczne eksperymenty, ponieważ automatyzacja zmniejsza ryzyko subiektywnych błędów i pominięć.

Jedną z największych zalet Auto-sklearn jest **automatyczne tworzenie zespołów modeli** (ensemble learning), które łączą najlepsze algorytmy w celu uzyskania lepszej ogólnej wydajności. Ensemble learning jest techniką, która w tradycyjnym podejściu wymagała dodatkowej pracy od specjalistów, ale Auto-sklearn integruje ją automatycznie.

#### Przykład porównania wyników

Poniżej przedstawiono porównanie wyników modelu utworzonego ręcznie i modelu stworzonego za pomocą Auto-sklearn. Ręczny model to klasyfikator **Random Forest**, a Auto-sklearn automatycznie dobiera najlepszy algorytm.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import autosklearn.classification

# Załadowanie danych
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ręczny model Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'Dokładność Random Forest: {accuracy_score(y_test, y_pred_rf):.4f}')

# Model Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300)
automl.fit(X_train, y_train)
y_pred_automl = automl.predict(X_test)
print(f'Dokładność Auto-sklearn: {accuracy_score(y_test, y_pred_automl):.4f}')
```

W tym przypadku, Auto-sklearn automatycznie dobiera różne algorytmy i może osiągnąć lepsze wyniki niż ręcznie utworzony model Random Forest, dzięki optymalizacji i ensemblingowi.

### 4.4. Zmniejszenie ryzyka błędów ludzkich w procesie modelowania

W tradycyjnym podejściu do modelowania, ręczne wybieranie algorytmów i dobieranie hiperparametrów naraża proces na **błędy ludzkie**. Specjaliści ds. danych mogą przypadkowo pomijać najlepsze algorytmy lub optymalizować modele w sposób suboptymalny. Auto-sklearn eliminuje ten problem, automatyzując cały proces.

Dzięki temu:
- **Zmniejsza się ryzyko błędnych wyborów algorytmów** – Auto-sklearn testuje wiele algorytmów, zamiast polegać na intuicji eksperta.
- **Optymalizacja hiperparametrów jest dokładniejsza** – Zamiast manualnego dobierania hiperparametrów, Auto-sklearn stosuje techniki takie jak optymalizacja bayesowska, które automatycznie dostosowują parametry do danych.

Dzięki automatyzacji, Auto-sklearn pozwala uniknąć wielu pułapek, które mogłyby prowadzić do suboptymalnych modeli w tradycyjnym podejściu.

#### Przykład zmniejszenia ryzyka błędów ludzkich

W poniższym przykładzie, użytkownik wybiera ręcznie algorytm z przypadkowym zestawem hiperparametrów, co prowadzi do słabszych wyników w porównaniu do automatycznego modelu.

```python
from sklearn.svm import SVC

# Ręczny wybór algorytmu SVM z losowymi hiperparametrami
svm = SVC(kernel='poly', C=100, gamma='auto')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f'Dokładność SVM: {accuracy_score(y_test, y_pred_svm):.4f}')

# Model Auto-sklearn automatycznie dobiera najlepsze hiperparametry
automl.fit(X_train, y_train)
y_pred_automl = automl.predict(X_test)
print(f'Dokładność Auto-sklearn: {accuracy_score(y_test, y_pred_automl):.4f}')
```

Wynik pokazuje, że automatyczne dostosowanie algorytmów i hiperparametrów prowadzi do lepszych rezultatów w porównaniu do manualnego procesu, co zmniejsza ryzyko błędów.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
