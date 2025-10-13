import pandas as pd

waves = ['k','l','n']
frames = []
for w in waves:
    df = pd.read_spss(f"../data/raw/UKHLS/{w}_indresp.sav")
    df = df.rename(columns={
        'pidp':'pidp', 'sex_dv':'sex', 'age_dv':'age',
        'gor_dv':'region', 'gamble_pgsi':'pgsi', 'ghq12score':'ghq12'
    })
    df['wave'] = w
    frames.append(df[['pidp','sex','age','region','pgsi','ghq12','wave']])

ukhls = pd.concat(frames, ignore_index=True)
ukhls.to_csv("../data/processed/ukhls_combined.csv", index=False)
print ("Data UKHLS combined and saved.")