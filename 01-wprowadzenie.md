## 1. Wprowadzenie do Auto-sklearn

### 1.1. Czym jest Auto-sklearn?

**Auto-sklearn** to narzędzie typu AutoML (Automated Machine Learning), które automatyzuje procesy związane z budowaniem i optymalizowaniem modeli uczenia maszynowego. Powstało na bazie popularnej biblioteki **scikit-learn** i działa jako nadbudowa, wykorzystując szeroką gamę algorytmów uczenia maszynowego dostępnych w tej bibliotece. Główną ideą Auto-sklearn jest umożliwienie użytkownikom tworzenia modeli bez konieczności ręcznego wybierania algorytmów, dobierania hiperparametrów, ani budowania zespołów modeli (ensemble).

Auto-sklearn optymalizuje cały proces modelowania, w tym:
- **Wybór algorytmów**: Automatycznie testuje różne algorytmy klasyfikacji lub regresji dostępne w scikit-learn.
- **Optymalizacja hiperparametrów**: Poprzez techniki wyszukiwania, takie jak **Bayesian Optimization**, dobiera odpowiednie hiperparametry modeli.
- **Meta-uczenie**: Wykorzystuje wcześniejsze doświadczenia zdobyte na innych zadaniach (dostępne w bazie wyników Auto-sklearn) do przyspieszenia optymalizacji.
- **Automatyczne tworzenie zestawu modeli**: Tworzy model finalny w formie **ensemblingu** kilku modeli o najlepszych wynikach, co pozwala na uzyskanie lepszej ogólnej wydajności.

### 1.2. Krótka historia i motywacja stojąca za stworzeniem Auto-sklearn

Auto-sklearn został stworzony przez **Matta Feurer** i jego współpracowników z Uniwersytetu w Freiburg w Niemczech. Narzędzie zostało po raz pierwszy zaprezentowane na **International Conference on Machine Learning (ICML)** w 2015 roku i szybko zyskało popularność w świecie nauki i przemysłu.

Motywacją za stworzeniem Auto-sklearn była potrzeba automatyzacji złożonych i czasochłonnych procesów związanych z tworzeniem modeli uczenia maszynowego. Budowa modeli często wymaga:
- **Eksperymentowania z różnymi algorytmami**,
- **Manualnego dobierania hiperparametrów**, co jest zadaniem czasochłonnym i podatnym na błędy,
- **Zaawansowanej wiedzy** na temat algorytmów, która może być przeszkodą dla mniej doświadczonych użytkowników.

Twórcy Auto-sklearn zauważyli, że nawet doświadczeni specjaliści popełniają błędy, ponieważ ręczna optymalizacja modeli jest trudna i podatna na subiektywne decyzje. Celem Auto-sklearn było zatem:
1. Zmniejszenie bariery wejścia dla użytkowników z ograniczoną wiedzą techniczną.
2. Skrócenie czasu potrzebnego do uzyskania wydajnych modeli.
3. Zwiększenie dokładności i stabilności modeli poprzez automatyczne dobieranie i optymalizację algorytmów.

Auto-sklearn jest kontynuacją tradycji narzędzi do automatyzacji modelowania, takich jak **Auto-WEKA**, ale jego zaletą jest znacznie szersza baza algorytmów i bardziej zaawansowane metody optymalizacji hiperparametrów.

### 1.3. Porównanie z innymi narzędziami AutoML

Auto-sklearn wyróżnia się na tle innych narzędzi AutoML dzięki kilku kluczowym funkcjom, ale ma również pewne ograniczenia. Oto porównanie Auto-sklearn z popularnymi narzędziami AutoML:

1. **TPOT** (Tree-based Pipeline Optimization Tool):
   - **Podobieństwa**: TPOT i Auto-sklearn są narzędziami opartymi na scikit-learn, które automatyzują proces wyboru modelu i optymalizacji hiperparametrów.
   - **Różnice**: TPOT opiera się na algorytmach ewolucyjnych do optymalizacji modelu, natomiast Auto-sklearn wykorzystuje **optymalizację bayesowską** i **meta-uczenie**, co przyspiesza proces optymalizacji. TPOT nie oferuje automatycznego tworzenia zestawów modeli (ensemble), co jest jedną z kluczowych funkcji Auto-sklearn.

2. **H2O AutoML**:
   - **Podobieństwa**: Oba narzędzia automatyzują proces budowy modeli i oferują automatyczne tworzenie zestawów modeli.
   - **Różnice**: H2O AutoML obsługuje zarówno dane wierszowe, jak i tablicowe, podczas gdy Auto-sklearn skupia się na danych tablicowych. H2O jest również szybsze, ponieważ jest zoptymalizowane do pracy na dużych zbiorach danych. Auto-sklearn ma jednak przewagę w postaci wykorzystania meta-uczenia oraz lepszej integracji z ekosystemem scikit-learn.

3. **Google Cloud AutoML**:
   - **Podobieństwa**: Oba narzędzia są zaprojektowane z myślą o użytkownikach, którzy chcą automatycznie trenować modele ML bez głębokiej wiedzy na temat algorytmów.
   - **Różnice**: Google Cloud AutoML to narzędzie komercyjne, dostępne w chmurze, które wymaga subskrypcji, podczas gdy Auto-sklearn jest open-source i można je uruchamiać lokalnie. Google Cloud AutoML ma również wbudowaną integrację z innymi usługami Google, co czyni je bardziej elastycznym w przypadku aplikacji w chmurze, ale mniej dostępnym dla indywidualnych użytkowników.

### Przykładowy kod Auto-sklearn

Poniżej znajduje się przykładowy kod demonstrujący, jak można używać Auto-sklearn do automatycznego tworzenia modelu klasyfikacyjnego na zbiorze danych **Iris**.

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Załadowanie zestawu danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utworzenie obiektu Auto-sklearn dla klasyfikacji
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30)

# Trening modelu Auto-sklearn na zbiorze treningowym
automl.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = automl.predict(X_test)

# Wyświetlenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')

# Wyświetlenie szczegółów dotyczących najlepszego modelu
print(automl.show_models())

# Wyświetlenie statystyk dotyczących procesu optymalizacji
print(automl.sprint_statistics())
```

### Wyjaśnienie kodu

1. **Import bibliotek**: Kod importuje moduł **autosklearn.classification** oraz klasyczne narzędzia z scikit-learn do ładowania zbiorów danych i ewaluacji modeli.
2. **Załadowanie danych**: Zestaw danych **Iris** to klasyczny zbiór danych do klasyfikacji, zawierający informacje o trzech gatunkach irysów.
3. **Podział na zbiór treningowy i testowy**: Dane są dzielone na dwie części: dane treningowe (80%) i dane testowe (20%).
4. **Trening Auto-sklearn**: Tworzymy obiekt **AutoSklearnClassifier**, który automatycznie wybiera algorytmy, optymalizuje hiperparametry i trenuje model przez maksymalnie 60 sekund.
5. **Predykcja**: Model predykcyjny jest stosowany do przewidywania klas dla zestawu testowego.
6. **Ewaluacja**: **accuracy_score** oblicza dokładność modelu, a następnie wyświetlamy szczegóły dotyczące najlepszego modelu i statystyki optymalizacji.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
