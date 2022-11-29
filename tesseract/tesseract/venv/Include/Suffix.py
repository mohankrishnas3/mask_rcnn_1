from collections import Counter
from ast import literal_eval
filter_list=['[5, 6, 7]', '[]', '[5, 6, 7]', '[]', '[5, 6, 7]', '[]', '[2, 3, 4, 5, 6, 7, 5, 6, 7]', '[3, 4, 5, 6, 7, 5, 6, 7]', '[5, 6, 7]', '[]', '[5, 6, 7]', '[]', '[5, 6, 7]', '[]']

list1= [literal_eval(x) for x in filter_list]
print(list1)
list2 = [x for x in list1 if x]
print(list2)
c = Counter(map(tuple,list2))
dups = [k for k,v in c.items() if v>1]
print(dups)
