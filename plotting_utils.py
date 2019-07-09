import matplotlib.pyplot as plt

def plot_model_qc(trend_array_training,trend_array_validation,trend_name):
    plt.title('Traning and Validation {}'.format(trend_name))
    plt.plot(range(len(trend_array_training)),trend_array_training,'bo',label='Training {}'.format(trend_name))
    plt.plot(range(len(trend_array_validation)),trend_array_validation,'b',label='Validation {}'.format(trend_name))
    plt.legend()

def new_figure():
    plt.figure()

def show_plot():
    plt.show()