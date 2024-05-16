# **Assingment 1**

**Description**:    
You own a web site that random users access for news. Your goal is to choose a news article to show to each user, that will maximize the chance that the user clicks on it (to read it further). This is also known as "the clickthrough rate". This is your problem setup more formally:

**News Articles**: 

When a user arrives at your site, there are a total of K=5 news articles you can choose from.
If you recommend article i then there is a probability that the user clicks article i, which is unknown and equal to pi.
Assume you have a total of T rounds, during which you want to maximize the number of successful recommendations (i.e., clicks)

**Users**:

For every user that visits your site, you know if they are: (i) male or female, and (ii) under or over 25 years old. 
The "characteristics" of each new user visiting your site, are drawn in an IID manner (i.e., the next user has no dependence on who the previous user was).
For your simulations, assume initially that there's an equal probability to draw a user from any of these classes. You can later compare this also with one scenario where these probabilities are not equal. (In all cases, assume you don't know these probabilities, beforehand, either).

**User-News Preference**: 

Unlike the standard bandits we've seen, it turns out that different types of users might prefer different articles! 
Let p1, p2, p3, p4 ,p5  denote the click probabilities for articles, 1,2,3,4,5, respectively. 

The taste differences are captured as follows:

female over 25: p1 = 0.8, p2 = 0.6, p3 = 0.5 , p4 = 0.4 , p5 = 0.2 

male over 25:  p1 = 0.2, p2 = 0.4, p3 = 0.5 , p4 = 0.6 , p5 = 0.8

male or female under 25:   p1 = 0.2, p2 = 0.4, p3 = 0.8 , p4 = 0.6 , p5 = 0.5

NOTE: Your algorithm initially knows NEITHER the ranking of different articles (per category), NOR the exact click probabilities.
It doesn't even know that males and females under 25 have similar preferences
