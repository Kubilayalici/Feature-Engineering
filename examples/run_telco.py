import pandas as pd
from pathlib import Path
from fe.telco import telco_basic_pipeline


def main():
    csv = Path(__file__).resolve().parents[1] / 'Telco_Customer_Churn_Feature_Engineering' / 'Telco-Customer-Churn.csv'
    df = pd.read_csv(csv)
    out = telco_basic_pipeline(df, drop_cols=['customerID'])
    print('Processed shape:', out.shape)
    print('Columns:', len(out.columns))


if __name__ == '__main__':
    main()

