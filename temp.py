# Python program to generate WordCloud 

# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 

# Reads 'Youtube04-Eminem.csv' file 
df = pd.read_csv(r"D:/reserch/law casess/1.txt", encoding ="utf-8") 
lines = df.read().splitlines()

comment_words = '' 
stopwords = set(STOPWORDS) 

# iterate through the csv file 
#for docs in lines: 
	
	# typecaste each val to string 
	#val = str(val) 

	# split the value 
	#tokens = val.split() 
	
	# Converts each token into lowercase 
for i in range(len(lines)): 
    lines[i] = lines[i].lower() 
    comment_words += " ".join(lines)+" "

wordcloud = WordCloud(width = 800, height = 800, 
				background_color ='white', 
				stopwords = stopwords, 
				min_font_size = 10).generate(comment_words) 

# plot the WordCloud image					 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 
