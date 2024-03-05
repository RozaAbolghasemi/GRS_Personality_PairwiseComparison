# A personality-aware group recommendation system based on pairwise preferences
Python code for a project focusing on "Group Recommendation Systems (GRS)" based on the personality of the users and their pairwise preferences on items.
"A personality-aware group recommendation system based on pairwise preferences (Information Science 2022)"



### project Description
Human personality plays a crucial role in decision-making and it has paramount importance when individuals negotiate with each other to reach a common group decision.
Such situations are conceivable, for instance, when a group of individuals want to watch
a movie together. It is well known that people influence each other’s decisions, the more
assertive a person is, the more influence they will have on the final decision. In order to
obtain a more realistic group recommendation system (GRS), we need to accommodate
the assertiveness of the different group members’ personalities. Although pairwise preferences are long-established in group decision-making (GDM), they have received very little
attention in the recommendation systems community. Driven by the advantages of pairwise preferences on ratings in the recommendation systems domain, we have further pursued this approach in this paper, however, we have done so for GRS. We have devised a
three-stage approach to GRS in which we 1) resort to three binary matrix factorization
methods, 2) develop an influence graph that includes assertiveness and cooperativeness
as personality traits, and 3) apply an opinion dynamics model to reach a consensus.
We have shown that the final opinion is related to the stationary distribution of a Markov
chain associated with the influence graph. Our exp

The logical diagram illustrating the process of the proposed method is as follows:
<p align="center">
<img style="width: 60%;" src="https://github.com/RozaAbolghasemi/GRS_Personality_PairwiseComparison/blob/main/Images/Diagram.png">
</p>

## Execution Dependencies
The codes can be run directly.
Also, the python code can be run by: 
```
python ./MFP.py
```

We are using pandas, numpy, scipy and warnings modules. Install them by
running.
```
pip install numpy
pip install pandas
pip install matplotlib.pyplot
```
The hyperparameters for matrix factorization, group sizes, and no. of generated groups can be changed through the config file.

### Dataset
* Pairwise preference data: The dataset for the MFP method was acquired from an online experiment performed by [Blèdaitè et al.](https://dl.acm.org/doi/pdf/10.1145/2700171.2791049?casa_token=hjYzq9yecUsAAAAA:oR_T8e6uKVasBZ77VpqAGnzFi0jRk__jeiz9DkGq3ZTQa3TSIjiii_zfJBSidmQ5LM4PDhHqMw_i) to collect users’
pairwise preferences. The authors developed an online interface that allows users to compare different movie pairs and enter
their pairwise scores. In this experiment, a total of 2,262 pairwise scores related to 100 movies from the MovieLens dataset
were collected based on feedback from 46 users. In addition, 73,078 movie ratings from 1,128 users in the [MovieLens 100 K](https://grouplens.org/datasets/movielens/100k/)
dataset were used. These movie ratings were converted into pairwise scores. 
<p align="center">
<img style="width: 60%;" src="https://github.com/RozaAbolghasemi/GRS_Personality_PairwiseComparison/blob/main/Images/Data_Pairwise.png">
</p>
----------------------------------------------------------------------

**License**

[MIT License](https://github.com/RozaAbolghasemi/GRS_Personality_PairwiseComparison/blob/main/LICENSE)


----------------------------------------------------------------------

**Reference**

If you use this code/paper, please cite it as below.
```
@article{abolghasemi2022personality,
  title={A personality-aware group recommendation system based on pairwise preferences},
  author={Abolghasemi, Roza and Engelstad, Paal and Herrera-Viedma, Enrique and Yazidi, Anis},
  journal={Information Sciences},
  volume={595},
  pages={1--17},
  year={2022},
  publisher={Elsevier}
}
```
