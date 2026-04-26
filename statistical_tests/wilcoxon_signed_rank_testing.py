from scipy.stats import wilcoxon

# f1 is used as a placeholder in the variable name, but this test was used to evaluate all metrics
proposed_f1 = [0.8275862068965517, 0.9137931034482759, 0.896551724137931, 0.896551724137931, 0.9137931034482759, 0.896551724137931, 0.8275862068965517, 0.8620689655172413, 0.9137931034482759, 0.896551724137931]
baseline_f1 = [0.7068965517241379, 0.5517241379310345, 0.603448275862069, 0.39655172413793105, 0.5517241379310345, 0.4482758620689655, 0.4482758620689655, 0.4827586206896552, 0.5689655172413793, 0.5]

stat, p_value = wilcoxon(baseline_f1, proposed_f1, alternative='two-sided')

print(f"P-value: {p_value}")

if p_value <= 0.05:
    print("Conclusion: Reject H0. The models are statistically significantly different!")
else:
    print("Conclusion: Fail to reject H0. The models are essentially the same.")