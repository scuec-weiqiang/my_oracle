from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 文泉驿微米黑字体路径（根据您的系统实际路径）
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
myfont = FontProperties(fname=font_path)

plt.figure()
# 在绘图时单独指定
plt.title("甲骨文识别", fontproperties=myfont)
plt.show()