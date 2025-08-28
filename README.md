# Feature Engineering (Telco Churn)

Bu repo, Telco Customer Churn verisi uzerinde ozellik muhendisligi adimlarini derli toplu hale getirmek icindir. Monolitik notebook/script yerine, yeniden kullanilabilir fonksiyonlar iceren bir `src/fe` paketi eklendi.

## Yapilanlar
- `src/fe/preprocess.py`: sutun secimi (`grab_col_names`), eksik deger tablosu, one-hot, binary label-encode, aykiri deger yaklasimlari.
- `src/fe/telco.py`: Telco’ ya ozel turetilmis ozellikler (`NEW_TENURE_YEAR`, `NEW_Engaged`, `NEW_TotalServices` vs.) ve basit bir pipeline (`telco_basic_pipeline`).
- `examples/run_telco.py`: CSV’yi isleyip nihai feature set’ini olusturan basit ornek.

## Hizli Baslangic

```bash
python examples/run_telco.py
```

## Kullanim (Kod)

```python
import pandas as pd
from fe.telco import telco_basic_pipeline

df = pd.read_csv('Telco_Customer_Churn_Feature_Engineering/Telco-Customer-Churn.csv')
Xy = telco_basic_pipeline(df, drop_cols=['customerID'])
```

## Notlar
- CatBoost gibi modeller one-hot/binary encode ile dogrudan calisir.
- Eger farkli veri setleri icin kullanmak isterseniz, `telco.py` yapisindaki gibi domain’e ozel fonksiyon yazip `preprocess.py` yardimcilariyla birlestirebilirsiniz.
