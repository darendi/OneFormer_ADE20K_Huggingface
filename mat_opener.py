import scipy.io
mat = scipy.io.loadmat('/home/darendy/OneFormer_Mapillary/color150.mat')
import pandas as pd

print(mat.keys())

x=mat['colors']
print(x)

# df=pd.DataFrame(x)
# df.to_csv('/home/darendy/OneFormer_Mapillary/data.csv', index = False)




