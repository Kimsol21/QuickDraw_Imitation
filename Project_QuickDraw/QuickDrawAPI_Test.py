from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup

qd = QuickDrawData()
anvil = qd.get_drawing("anvil")

# print(anvil)

'''
print(anvil.name)
print(anvil.key_id)
print(anvil.countrycode)
print(anvil.recognized)
print(anvil.timestamp)
print(anvil.no_of_strokes)
print(anvil.image_data)
print(anvil.strokes)
'''

# anvil.image.save("my_anvil.gif")
# anvil.animation.save("my_anvil_animation.gif")

'''
anvils = QuickDrawDataGroup("anvil")
print(anvils.drawing_count)
print(anvils.get_drawing())
'''

#anvils = QuickDrawDataGroup("anvil", max_drawings=None)
#print(anvils.drawing_count)

#qdg = QuickDrawDataGroup("anvil")
#for drawing in qdg.drawings:
#    print(drawing)

qd = QuickDrawData()
print(qd.drawing_names)


