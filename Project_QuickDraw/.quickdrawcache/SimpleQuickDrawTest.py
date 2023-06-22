#데이터 로드
# define 10 classes to load the data for
categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']
label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
                      5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword'}

# load data for each category
classes = {}
for category in categories:
    data = pd.read_csv("../input/train_simplified/" + category + ".csv")
    classes[category] = data