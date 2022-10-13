import dill
import copy

queries = ['InTopK', 'NotInTopK', 'WhyThisTopK', 'WhyInTheseTopKs']
for functions in ['L', 'NL']:
  for q in queries:
    for dataset in ['A', 'C', 'I']:
      filename = "Synthetic" + dataset + 'Z' + functions + "-Clean.dill"
      data = dill.load(open(filename, 'rb'))
      str_query = "M" + dataset + 'Z' + functions + q 
      brute = ','.join([str(data[q]['BruteForce']['RuntimeNS']/10**9)]*len(data[q]['Approximate'].keys()))
      print(str_query + "BRUTE = [" + brute + "];")
      print("plot(x(1:length(" + str_query + "BRUTE )), " + str_query + "BRUTE,'LineWidth',2.0);")
      print("hold on;")
      approx = ','.join(list(map(lambda x: str(data[q]['Approximate'][x]['RuntimeNS']/10**9), sorted(data[q]['Approximate'].keys()))))
      print(str_query + "Approximate = [" + approx + "];")
      print("plot(x(1:length(" + str_query + "Approximate  )), " + str_query + "Approximate,'LineWidth',2.0);")
      print("hold on;")
      shap = ','.join(list(map(lambda x: str(data[q]['SHAP'][x]['RuntimeNS']/10**9), sorted(data[q]['SHAP'].keys()))))
      print(str_query + "KernelSHAP  = [" + shap + "];")
      print("plot(x(1:length(" + str_query + "KernelSHAP )), " + str_query + "KernelSHAP ,'LineWidth',2.0);")
      print("hold on;")
      #print()
    print()
    print()
  print()
  print()
