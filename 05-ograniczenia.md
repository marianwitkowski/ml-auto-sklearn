## 5. Ograniczenia i wyzwania Auto-sklearn

Mimo wielu zalet, Auto-sklearn ma pewne ograniczenia i wyzwania, które należy wziąć pod uwagę podczas jego używania. Te ograniczenia są istotne zarówno dla początkujących użytkowników, jak i doświadczonych specjalistów ds. danych, ponieważ mogą one wpływać na efektywność narzędzia w zależności od konkretnego zastosowania. W tej sekcji omówimy kluczowe problemy związane ze skalą i wydajnością, interpretowalnością wyników, predefiniowanymi algorytmami oraz wymaganiami zasobów obliczeniowych.

### 5.1. Skala i wydajność na dużych zbiorach danych

Jednym z głównych wyzwań związanych z Auto-sklearn jest **skala i wydajność** narzędzia na dużych zbiorach danych. Auto-sklearn został zaprojektowany do pracy z zestawami danych o umiarkowanej wielkości, ale na bardzo dużych zestawach danych jego wydajność może drastycznie spadać. 

Dzieje się tak z kilku powodów:
1. **Optymalizacja hiperparametrów** – Proces optymalizacji hiperparametrów (np. za pomocą optymalizacji bayesowskiej) może wymagać dużej liczby prób i obliczeń, co powoduje wydłużenie czasu przetwarzania na dużych zbiorach danych.
2. **Testowanie wielu algorytmów** – Auto-sklearn testuje wiele algorytmów i ich różnych konfiguracji. Na dużych zbiorach danych czas potrzebny do przetworzenia każdej z tych konfiguracji może być bardzo długi.
3. **Ensemble learning** – Tworzenie zespołów modeli wymaga dodatkowego czasu i zasobów obliczeniowych, co może być problematyczne na bardzo dużych zestawach danych.

#### Przykład

Na dużych zbiorach danych, takich jak miliony rekordów i setki cech, używanie Auto-sklearn może nie być optymalne bez odpowiednich zasobów. W takich przypadkach warto rozważyć techniki przetwarzania danych takie jak **próbkowanie danych** (sampling), aby zmniejszyć ich rozmiar i przyspieszyć działanie algorytmów.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import autosklearn.classification

# Tworzenie dużego zbioru danych (np. 1 milion próbek)
X, y = make_classification(n_samples=1000000, n_features=20, random_state=42)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja Auto-sklearn (może to potrwać długo na dużych zbiorach danych)
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600)
automl.fit(X_train, y_train)

# Predykcja
y_pred = automl.predict(X_test)
```

W przypadku bardzo dużych zbiorów danych, czas przetwarzania może sięgać godzin lub dni, zwłaszcza jeśli ograniczone są zasoby obliczeniowe.

### 5.2. Interpretowalność wyników

Drugim istotnym ograniczeniem Auto-sklearn jest **interpretowalność wyników**. Modele automatycznie generowane przez Auto-sklearn są często złożonymi zespołami algorytmów (ensembling), co utrudnia zrozumienie, dlaczego dany model podejmuje określone decyzje.

W dziedzinach takich jak medycyna, finanse czy prawo, gdzie transparentność decyzji modelu jest kluczowa, interpretowalność jest jednym z najważniejszych aspektów. Modele typu „black box” (czarne skrzynki), generowane przez narzędzia takie jak Auto-sklearn, mogą być trudne do wyjaśnienia.

#### Sposoby na poprawę interpretowalności

Jednym ze sposobów radzenia sobie z tym ograniczeniem jest zastosowanie narzędzi do interpretacji modeli, takich jak:
- **SHAP (Shapley Additive Explanations)** – metoda wyjaśniania wyników modeli oparta na teorii gier.
- **LIME (Local Interpretable Model-agnostic Explanations)** – narzędzie do wyjaśniania wyników modeli w sposób lokalny.

Przykład wyjaśniania predykcji za pomocą SHAP:

```python
import shap

# Inicjalizacja explainer SHAP dla modelu
explainer = shap.Explainer(automl.predict, X_train)
shap_values = explainer(X_test)

# Wizualizacja wyjaśnień
shap.summary_plot(shap_values, X_test)
```

Używając narzędzi takich jak SHAP, można lepiej zrozumieć, które cechy wpływają na decyzje modelu. Jednak dodatkowe narzędzia mogą zwiększyć złożoność całego procesu i nie zawsze rozwiązują problem interpretacji w pełni złożonych modeli.

### 5.3. Wyzwania związane z predefiniowanymi algorytmami

Auto-sklearn opiera się głównie na algorytmach dostępnych w bibliotece **scikit-learn**, co oznacza, że użytkownicy są ograniczeni do predefiniowanego zestawu algorytmów i ich hiperparametrów. Choć scikit-learn oferuje szeroki wachlarz algorytmów, w niektórych przypadkach bardziej zaawansowane lub specjalistyczne algorytmy mogą być konieczne, a te nie są obsługiwane przez Auto-sklearn.

#### Przykłady:

1. **Sieci neuronowe** – Auto-sklearn nie obsługuje głębokich sieci neuronowych, takich jak te oferowane przez biblioteki TensorFlow lub PyTorch. Dla zaawansowanych zadań, takich jak rozpoznawanie obrazów lub przetwarzanie języka naturalnego, sieci neuronowe mogą być bardziej efektywne niż algorytmy dostępne w Auto-sklearn.
   
2. **Brak algorytmów specyficznych dla niektórych dziedzin** – W dziedzinach takich jak analiza szeregów czasowych, algorytmy takie jak LSTM (Long Short-Term Memory) mogą być bardziej odpowiednie, jednak Auto-sklearn ich nie wspiera.

#### Przykład braku wsparcia dla algorytmów:

```python
# Próba użycia sieci neuronowej (Auto-sklearn nie obsługuje takich algorytmów)
from sklearn.neural_network import MLPClassifier

automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300)
# Brak wsparcia dla zaawansowanych algorytmów spoza scikit-learn
```

Rozwiązaniem może być kombinacja Auto-sklearn z innymi narzędziami, ale może to wymagać dodatkowego wysiłku w integracji różnych technologii.

### 5.4. Wymagania zasobów obliczeniowych

Ostatnim ograniczeniem, które warto omówić, są **wymagania zasobów obliczeniowych**. Auto-sklearn jest narzędziem intensywnie korzystającym z mocy obliczeniowej. Automatyzacja testowania wielu algorytmów i optymalizacji hiperparametrów może wymagać znacznej ilości zasobów, takich jak procesory (CPU), pamięć RAM oraz czas obliczeń.

Na przeciętnym komputerze może być trudno trenować duże modele lub optymalizować hiperparametry na większych zbiorach danych. Auto-sklearn może również zająć dużo pamięci RAM, zwłaszcza gdy przetwarza zestawy danych o dużej liczbie cech.

#### Przykład:

Jeśli użytkownik korzysta z laptopa lub maszyny z ograniczonymi zasobami, Auto-sklearn może nie być w stanie efektywnie przeprowadzić swoich obliczeń. W takich przypadkach dobrym rozwiązaniem jest skorzystanie z zasobów chmurowych, takich jak AWS lub Google Cloud, aby uruchamiać Auto-sklearn na mocniejszych instancjach obliczeniowych.

```python
# Przykład wywołania Auto-sklearn z limitem czasu i liczby procesów (dla maszyn o mniejszych zasobach)
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=600,
    per_run_time_limit=120,
    n_jobs=2  # Ograniczenie do 2 procesów równoległych
)
```

W zależności od zasobów, użytkownik może dostosować liczbę równoległych procesów (**n_jobs**) i czas przetwarzania dla każdego modelu. Ograniczenia te są szczególnie istotne, gdy zasoby sprzętowe są niewystarczające.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
