# cct-midterm

I used the Cultural Consensus Theory (CCT) model in PyMC to analyze responses from 10 informants who answered 20 yes/no questions about plant knowledge. The model estimated each informants’s competence (Di) and the shared consensus answers (Zj). I applied a shifted Beta(2,1) prior for competence to keep values between 0.5 and 1.0 while favoring higher competence, and used a non-informative Bernoulli(0.5) prior for consensus answers. MCMC sampling ran with 4 chains and 2,000 draws per chain, and convergence was excellent (all R-hat values = 1.000). Results showed variation in competence, with P6 being the most competent (0.869) and P3 the least (0.606), forming clear high, medium, and low competence groups.

The consensus answers matched simple majority voting on 75% of the questions. For the remaining 25%, the CCT model often sided with more competent informants, even when they were in the minority. This shows one of the key strengths of the model—it weighs informant responses based on estimated knowledge rather than treating all inputs equally. Posterior probabilities showed high confidence for most questions, except for a couple with more disagreement. Overall, the CCT approach provided more nuanced and likely more accurate consensus answers than simple aggregation.

(Project completed with AI assistance)
