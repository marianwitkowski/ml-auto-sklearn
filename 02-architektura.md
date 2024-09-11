## 2. Architektura i zasady działania Auto-sklearn

Auto-sklearn to potężna i złożona platforma AutoML, która automatyzuje wiele procesów związanych z budowaniem modeli uczenia maszynowego. Jej architektura została zaprojektowana w taki sposób, aby możliwie jak najbardziej uprościć cały proces modelowania, począwszy od wyboru algorytmu, przez optymalizację hiperparametrów, aż po tworzenie zestawu najlepszych modeli (ensemble). W tej części omówimy szczegóły architektury Auto-sklearn oraz procesy, które za nią stoją.

### 2.1. Ogólny przegląd architektury Auto-sklearn

Architektura Auto-sklearn opiera się na kilku kluczowych komponentach:

1. **Baza algorytmów**: Auto-sklearn korzysta z szerokiego zestawu algorytmów dostępnych w bibliotece **scikit-learn**. To właśnie na bazie tych algorytmów system buduje modele.
  
2. **Optymalizacja hiperparametrów**: Za pomocą zaawansowanych technik, takich jak optymalizacja bayesowska, Auto-sklearn dobiera optymalne hiperparametry dla każdego algorytmu.
  
3. **Meta-uczenie**: Auto-sklearn wykorzystuje dane z poprzednich zadań uczenia maszynowego, aby przyspieszyć proces optymalizacji na nowym zadaniu.
  
4. **Tworzenie zestawów modeli (ensemble)**: Po przeprowadzeniu optymalizacji, Auto-sklearn łączy najlepsze modele w zestaw, co zwiększa wydajność końcowego rozwiązania.

5. **Kontrola zasobów**: Auto-sklearn umożliwia kontrolowanie zasobów obliczeniowych, takich jak czas na zadanie i maksymalny czas trwania pojedynczej iteracji.

### 2.2. Proces automatyzacji modelowania

Auto-sklearn automatyzuje cały proces budowania modeli uczenia maszynowego. Proces ten można podzielić na kilka kluczowych etapów:

#### 2.2.1. Wybór algorytmów

W pierwszym etapie Auto-sklearn automatycznie wybiera różne algorytmy uczenia maszynowego spośród tych dostępnych w scikit-learn. Może to być np. **Random Forest**, **Support Vector Machines**, **Gradient Boosting**, **K-Nearest Neighbors** i wiele innych.

Wybór algorytmu odbywa się poprzez ocenę wielu algorytmów na podstawie wyników ich wstępnych prób. Auto-sklearn korzysta z historii poprzednich zadań (o ile dostępna jest baza meta-uczenia), aby skrócić czas potrzebny na wybór algorytmu. Dodatkowo, system iteracyjnie testuje i ocenia różne kombinacje algorytmów, aby znaleźć te, które najlepiej radzą sobie z danym zestawem danych.

#### 2.2.2. Optymalizacja hiperparametrów

Po wyborze odpowiednich algorytmów, Auto-sklearn przystępuje do optymalizacji ich hiperparametrów. Kluczowym elementem tego procesu jest **Bayesian Optimization** (optymalizacja bayesowska), która jest bardziej efektywna niż metody brute-force, takie jak **Grid Search** czy **Random Search**.

Bayesian Optimization działa na zasadzie iteracyjnej oceny możliwych zestawów hiperparametrów. Dla każdego nowego algorytmu tworzy model probabilistyczny, który przewiduje, jak dobrze dany zestaw hiperparametrów będzie działał. Z biegiem czasu ten model probabilistyczny staje się coraz bardziej precyzyjny, co pozwala systemowi szybciej znaleźć optymalne rozwiązania.

Przykładowy kod przedstawiający optymalizację modelu za pomocą Auto-sklearn:

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ładowanie danych
X, y = load_iris(return_X_y=True)

# Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja klasyfikatora Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)

# Trening modelu
automl.fit(X_train, y_train)

# Predykcja
y_pred = automl.predict(X_test)

# Ocena modelu
print(f'Dokładność: {accuracy_score(y_test, y_pred):.2f}')
```

W tym przykładzie Auto-sklearn automatycznie wybiera algorytm i optymalizuje jego hiperparametry. Kod ten jest bardzo prosty, ale pokazuje ogromną moc Auto-sklearn w automatyzacji złożonych procesów.

#### 2.2.3. Meta-uczenie (meta-learning)

Meta-uczenie to kluczowy element architektury Auto-sklearn, który pozwala na przyspieszenie procesu optymalizacji, korzystając z doświadczeń zdobytych na innych zestawach danych. Auto-sklearn korzysta z bazy wyników pochodzących z wcześniejszych eksperymentów, aby na ich podstawie dokonać bardziej efektywnego wyboru algorytmów i hiperparametrów.

Dzięki meta-uczeniu Auto-sklearn nie musi zaczynać od zera w każdej nowej sesji treningowej. Może "nauczyć się" z poprzednich zadań, co przyspiesza proces optymalizacji na nowych zadaniach, szczególnie gdy dane są podobne do tych, na których system był wcześniej trenowany.

Przykład: Jeśli Auto-sklearn ma w swojej bazie danych wiele zadań klasyfikacji na zbiorach danych o podobnych charakterystykach (np. dane tablicowe), może automatycznie priorytetyzować algorytmy i hiperparametry, które sprawdziły się w podobnych zadaniach.

#### 2.2.4. Ensemble learning

Jednym z głównych atutów Auto-sklearn jest automatyczne tworzenie zestawów modeli (ensemble learning). Proces ten polega na łączeniu kilku najlepszych modeli w jeden ostateczny model, co zazwyczaj zwiększa dokładność i stabilność predykcji.

Auto-sklearn stosuje techniki **baggingu** i **boostingu** w celu stworzenia zespołów modeli. Po przetestowaniu różnych kombinacji algorytmów i ich hiperparametrów, system wybiera te, które osiągają najlepsze wyniki i łączy je w ostateczny model.

Ensemble learning jest skuteczny, ponieważ różne modele mogą uczyć się różnych aspektów problemu, co zwiększa ogólną wydajność. Auto-sklearn automatycznie optymalizuje wagę przypisaną poszczególnym modelom w zestawie, aby uzyskać najlepsze wyniki.

### 2.3. Znaczenie i działanie wstępnego przetwarzania danych (preprocessing)

Przed przystąpieniem do modelowania, bardzo ważne jest wstępne przetwarzanie danych. Auto-sklearn automatyzuje również ten proces, który obejmuje:

1. **Skalowanie danych**: Wiele algorytmów uczenia maszynowego wymaga odpowiedniego przeskalowania cech, np. za pomocą standardyzacji czy normalizacji.
   
2. **Imputacja brakujących wartości**: Auto-sklearn automatycznie radzi sobie z brakującymi wartościami w danych, stosując odpowiednie techniki imputacji.

3. **Kategoryzacja cech**: Jeśli w danych znajdują się zmienne kategoryczne, Auto-sklearn automatycznie przekształca je na odpowiednie reprezentacje numeryczne, takie jak one-hot encoding.

Przykład zastosowania wstępnego przetwarzania w Auto-sklearn:

```python
import autosklearn.pipeline.components.data_preprocessing
from sklearn.preprocessing import StandardScaler

# Auto-sklearn automatycznie dobiera odpowiednie techniki preprocessingowe,
# ale można także użyć niestandardowych rozwiązań:
scaler = StandardScaler()

# Skalowanie danych
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Użycie Auto-sklearn na przetworzonych danych
automl.fit(X_train_scaled, y_train)
```

Dzięki automatyzacji wstępnego przetwarzania danych, użytkownicy nie muszą ręcznie dobierać metod przekształcania danych. To znacznie przyspiesza proces tworzenia modeli i sprawia, że Auto-sklearn jest bardziej wszechstronny.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
