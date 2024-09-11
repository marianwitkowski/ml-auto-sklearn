## 3. Kluczowe funkcjonalności Auto-sklearn

Auto-sklearn to jedno z najbardziej zaawansowanych narzędzi AutoML, które automatyzuje wiele procesów związanych z budową modeli uczenia maszynowego. W tej sekcji omówimy najważniejsze funkcjonalności Auto-sklearn, takie jak automatyczny wybór najlepszego modelu, automatyczna inżynieria cech, korzystanie z bibliotek scikit-learn, metody walidacji krzyżowej oraz meta-learning.

### 3.1. Automatyczny wybór najlepszego modelu

Jedną z najważniejszych funkcji Auto-sklearn jest automatyczny wybór najlepszego algorytmu uczenia maszynowego. Dzięki tej funkcji użytkownicy nie muszą ręcznie testować różnych algorytmów i porównywać ich wyników, co w tradycyjnym procesie budowy modeli jest czasochłonne i skomplikowane.

Auto-sklearn wybiera najlepszy algorytm na podstawie zdefiniowanych kryteriów, takich jak dokładność, f1-score, czy inne miary oceny modelu. W procesie automatyzacji, narzędzie wykorzystuje **Bayesian Optimization**, która iteracyjnie dostosowuje różne algorytmy i ich hiperparametry. Optymalizacja bayesowska zmniejsza liczbę niezbędnych prób i minimalizuje ryzyko wyboru niewłaściwego algorytmu, co jest ogromną przewagą nad bardziej tradycyjnymi metodami, takimi jak **grid search** czy **random search**.

#### Przykład kodu:

```python
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Załadowanie danych
X, y = load_breast_cancer(return_X_y=True)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja Auto-sklearn i wybór najlepszego modelu
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
automl.fit(X_train, y_train)

# Predykcja na zestawie testowym
y_pred = automl.predict(X_test)

# Ocena modelu
print(f'Dokładność: {accuracy_score(y_test, y_pred):.4f}')
```

W powyższym przykładzie Auto-sklearn automatycznie wybiera najlepszy model dla danych **breast cancer**. Cały proces jest zautomatyzowany, co znacznie ułatwia i przyspiesza pracę.

### 3.2. Automatyczna inżynieria cech

**Inżynieria cech** to proces przekształcania i tworzenia nowych cech na podstawie istniejących danych, co często prowadzi do poprawy wyników modeli. Auto-sklearn automatyzuje ten proces, stosując różne techniki takie jak **skalowanie**, **normalizacja** czy **kategoryzacja**. Dzięki tej funkcjonalności, użytkownicy nie muszą ręcznie wybierać i testować różnych transformacji cech.

Dodatkowo, Auto-sklearn posiada moduły do imputacji brakujących danych oraz transformacji zmiennych kategorycznych. Automatyczne przetwarzanie danych pomaga eliminować problemy wynikające z różnic w skali czy typach danych.

#### Przykład kodu:

```python
import autosklearn.pipeline.components.feature_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Załadowanie danych
X, y = load_iris(return_X_y=True)

# Inżynieria cech przy użyciu StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicjalizacja Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
automl.fit(X_train, y_train)
```

Tutaj pokazano, jak Auto-sklearn może automatycznie przeprowadzić inżynierię cech, ale można także ręcznie wprowadzić niektóre kroki (jak skalowanie), by lepiej kontrolować proces przetwarzania danych.

### 3.3. Korzystanie z bibliotek scikit-learn i optymalizacja parametrów

Jedną z największych zalet Auto-sklearn jest jego ścisła integracja z biblioteką **scikit-learn**, która jest standardem w świecie uczenia maszynowego. Auto-sklearn rozszerza funkcjonalność scikit-learn poprzez automatyzację wyboru algorytmów i optymalizację hiperparametrów.

Scikit-learn oferuje bogaty zestaw algorytmów i narzędzi, które Auto-sklearn może automatycznie zastosować do modelowania. Dzięki optymalizacji hiperparametrów, Auto-sklearn testuje różne kombinacje parametrów w celu znalezienia najlepszych ustawień dla konkretnego problemu.

#### Przykład kodu:

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import autosklearn.classification

# Załadowanie danych i podział na zbiory treningowe i testowe
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Uruchomienie Auto-sklearn z optymalizacją parametrów
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Predykcja i ocena
y_pred = automl.predict(X_test)
print(classification_report(y_test, y_pred))
```

W tym kodzie Auto-sklearn optymalizuje parametry, wybierając najlepsze kombinacje algorytmów scikit-learn, aby stworzyć najlepszy możliwy model dla danych **digits**.

### 3.4. Metody walidacji krzyżowej (cross-validation)

Auto-sklearn automatycznie wykorzystuje różne metody walidacji, takie jak **k-fold cross-validation**, aby ocenić modele w sposób dokładniejszy niż proste podziały na zbiory treningowe i testowe. Cross-validation polega na podziale danych na k równych części, gdzie każda z części jest na przemian używana jako zbiór testowy, a pozostałe k-1 jako zbiór treningowy. Auto-sklearn integruje te techniki, aby lepiej ocenić wydajność modelu i zminimalizować ryzyko **overfittingu** (przeuczenia).

#### Przykład kodu:

```python
from autosklearn.metrics import accuracy
import autosklearn.classification
from sklearn.model_selection import cross_val_score

# Załadowanie danych
X, y = load_breast_cancer(return_X_y=True)

# Inicjalizacja Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, resampling_strategy='cv', resampling_strategy_arguments={'folds': 5})

# Trening modelu z walidacją krzyżową
automl.fit(X, y)

# Wyświetlenie wyników walidacji
print(automl.sprint_statistics())
```

W tym przykładzie Auto-sklearn wykorzystuje 5-fold cross-validation, aby lepiej ocenić model przed ostatecznym treningiem.

### 3.5. Meta-learning: Uczenie na podstawie wcześniejszych modeli

Jedną z najbardziej zaawansowanych funkcji Auto-sklearn jest **meta-learning**. Jest to proces, w którym Auto-sklearn korzysta z wiedzy zgromadzonej w poprzednich zadaniach uczenia maszynowego, aby przyspieszyć optymalizację i wybór algorytmów w nowym zadaniu. Jeśli istnieją podobne zadania, które były wcześniej rozwiązywane, Auto-sklearn może skorzystać z tych informacji, aby lepiej wybrać algorytmy i hiperparametry.

Meta-learning jest szczególnie użyteczny, gdy dane wejściowe mają podobną strukturę lub charakterystykę do tych, które były wcześniej analizowane. W ten sposób Auto-sklearn może redukować czas treningu i poprawiać wydajność, wykorzystując wcześniejsze doświadczenia.

#### Przykład kodu:

```python
from autosklearn.metalearning.metalearning import MetaLearning
import autosklearn.classification

# Inicjalizacja meta-learningu
meta_learner = MetaLearning()

# Przykładowe zastosowanie w Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300)
automl.fit(X_train, y_train)

# Predykcja
y

_pred = automl.predict(X_test)
```

W powyższym kodzie, Auto-sklearn korzysta z wiedzy meta-learningu do optymalizacji, co przyspiesza proces budowy modelu.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
