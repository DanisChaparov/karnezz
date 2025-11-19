train["objects_per_customer"] = train["objects_count"] / (train["customers_count"]+1)
train["bill_per_object"] = train["avg_bill"] / (train["objects_count"]+1)
train["revenue_per_customer"] = train["transaction_sum"] / (train["customers_count"]+1)





for col in ["customers_count", "objects_count", "avg_bill"]:
    train[col+"_log"] = np.log1p(train[col])
    test[col+"_log"]  = np.log1p(test[col])
