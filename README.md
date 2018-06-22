# master_thesis
Privacy preserving machine learning

## Abstract



Distributed privacy preserving machine learning is a growing field which is gaining more relevance by the minute. The aim of distributed privacy preserving machine learning is to enable learning between multiple parties without actually sharing data. This can be very useful as it enables organization to collaborate. However, due to legislation as well as ethical and economic factors, organizations might be inhibited from sharing data. The methods discussed in this thesis might help organizations overcome these issues.




Multiple frameworks exit for achieving privacy preserving machine learning. This thesis examines how privacy and error rate are tied together. Three different algorithms for providing privacy were constructed. A distributed Lasso which only shares gradients between data centers, A regularized logistic regression with differential privacy and a stochastic gradient descent with differentially private updates.

The differentially private algorithms show that the amount of training data used to build the predictive model is important. Both mechanism are able to tie their sensitivity to the amount of training data, resulting in much better error rate for large amount of data. The stochastic gradient decent was able to show that using random projections beforehand can improve the error rate. Finally, the distributed Lasso showed that for a specific amount of data collaboration between parties can result in a strong improvement on the error rate.



