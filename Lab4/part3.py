# Lab 4 in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Anna Jonsson and Amanda Bigelius
# ---- PART 3 ----

# Dependencies
from summa.summarizer import summarize
from summa import keywords

print('\n LETS START PART 3 \n')

# Text from tutorial: 
text = 'Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organized representation in many countries and international networks. It borrows elements from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism and initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries (especially Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to the ideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia, anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism.'

# Define length of the summary as a proportion of the text
print(summarize(text, ratio=0.2))

# Create summary by specifying the length as number of words
summarize(text, words = 50)

# Extract the most important words from the text
print('\nKeywords:\n', keywords.keywords(text))

# Print top 3 keywords
print('\nTop 3 Keywords:\n',keywords.keywords(text,words=3))

print('\n THE END OF PART 3 \n')