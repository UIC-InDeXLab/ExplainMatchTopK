
import dill

data = dill.load(open('VaryingMCandidates-Clean.dill', 'rb'))

queries = ['InTopK', 'NotInTopK', 'WhyThisTopK', 'WhyInTheseTopKs']
for q in queries:
    aprx_diff = list(map(lambda x: data[q]['Approximate'][x]['AverageDiffernce'], sorted(data[q]['Approximate'].keys())))
    aprx_dev = list(map(lambda x: data[q]['Approximate'][x]['StdDifference'], sorted(data[q]['Approximate'].keys())))
    shap_diff = list(map(lambda x: data[q]['SHAP'][x]['AverageDiffernce'], sorted(data[q]['SHAP'].keys())))
    shap_dev = list(map(lambda x: data[q]['SHAP'][x]['StdDifference'], sorted(data[q]['SHAP'].keys())))
    diff = list(zip(aprx_diff, shap_diff))
    dev = list(zip(aprx_dev, shap_dev))
    diff_str = ';'.join(list(map(lambda x: ','.join(list(map(str, x))), diff)))
    dev_str = ';'.join(list(map(lambda x: ','.join(list(map(str, x))), dev)))
    print("errorBar=[" + diff_str + "];")
    print("errorDev=[" + dev_str + "];")
    print()
    print()
