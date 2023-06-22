import os
from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup

qd = QuickDrawData()

mermaids = QuickDrawDataGroup("mermaid", max_drawings=1200)
pandas = QuickDrawDataGroup("panda", max_drawings=1200)

index=1
for mermaid in mermaids.drawings:
    mermaid.image.save("./data/mermaid/mermaid"+str(index)+".jpg")
    index=index+1
    print(index)

index=1
for panda in pandas.drawings:
    panda.image.save("./data/panda/panda"+str(index)+".jpg")
    index=index+1
    print(index)

