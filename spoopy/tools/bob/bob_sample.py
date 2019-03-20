"""
This code was taken from the Article "A Complete Application: Analysis of the Fisher Iris Dataset"
(https://www.idiap.ch/software/bob/docs/bob/bob/stable/example.html)

# """
# import bob.db.iris
# import bob.learn.linear
# import bob.measure
# import numpy
#
# from matplotlib import pyplot
#
# trainer = bob.learn.linear.FisherLDATrainer()
#
# data = bob.db.iris.data()
#
# # here we create a linear machine to perform LDA (Linear Discriminant Analysis)
# # on the Iris dataset using the created FisherLDATrainer (trainer)
# machine, unused_eigen_values = trainer.train(data.values())
# print(machine.shape)
#
# output = {}
#
# for key in data:
#     # output will contain the LDA projected information as a 2d numpy ndarray
#     output[key] = machine.forward(data[key])
#
# # visualization of the results with matplot
# pyplot.hist(output['setosa'][:, 0], bins=8, color='green', label='Setosa', alpha=0.5)
# pyplot.hist(output['versicolor'][:, 0], bins=8, color='blue', label='Versicolor', alpha=0.5)
# pyplot.hist(output['virginica'][:, 0], bins=8, color='red', label='Virginica', alpha=0.5)
#
# pyplot.legend()
# pyplot.grid(True)
# pyplot.axis([-3, +3, 0, 20])
# pyplot.title("Iris Plants / 1st. LDA component")
# pyplot.xlabel("LDA[0]")
# pyplot.ylabel("Count")
#
# pyplot.show()
#
# pyplot.gcf().clear()
#
# # measuring the performance
# negatives = numpy.vstack([output['setosa'], output['versicolor']])[:, 0]
# positives = output['virginica'][:, 0]
# threshold = bob.measure.eer_threshold(negatives, positives)
#
# true_rejects = bob.measure.correctly_classified_negatives(negatives, threshold)
# true_accepts = bob.measure.correctly_classified_positives(positives, threshold)
#
# sum_true_rejects = sum(true_rejects)
# sum_true_accepts = sum(true_accepts)
#
# print(sum_true_rejects)
# print(sum_true_accepts)
#
# # plot ROC (Receiver Operating Characteristic) curve
# bob.measure.plot.roc(negatives, positives)
# pyplot.xlabel("False Virginica Acceptance (%)")
# pyplot.ylabel("False Virginica Rejection (%)")
#
# pyplot.title("ROC curve for Virginica Classification")
# pyplot.grid()
# pyplot.axis([0, 5, 0, 15])
#
# pyplot.show()
