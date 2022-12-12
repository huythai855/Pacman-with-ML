import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        return self.w

    def run(self, x):
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        batch_size = 1
        continue_training = True
        while continue_training:
            continue_training = False
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    continue_training = True
                    self.w.update(
                        direction=nn.Constant(nn.as_scalar(y) * x.data), multiplier=1
                    )


class RegressionModel(object):

    def __init__(self):
        self.lr = 0.01
        self.w1 = nn.Parameter(1, 256)
        self.b1 = nn.Parameter(1, 256)

        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)

        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        
        self.w4 = nn.Parameter(64, 1)
        self.b4 = nn.Parameter(1, 1)
        self.params = [
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.w4,
            self.b4,
        ]

    def run(self, x):
        first_layer_out = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        second_layer_out = nn.ReLU(
            nn.AddBias(nn.Linear(first_layer_out, self.w2), self.b2)
        )
        third_layer_out = nn.ReLU(
            nn.AddBias(nn.Linear(second_layer_out, self.w3), self.b3)
        )
        last_layer_out = nn.AddBias(nn.Linear(third_layer_out, self.w4), self.b4)
        return last_layer_out

    def get_loss(self, x, y):
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        batch_size = 100
        loss = float("inf")
        while loss >= 0.015:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                print(nn.as_scalar(loss))
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

class DigitClassificationModel(object):

    def __init__(self):
        self.lr = 0.1
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        self.params = [
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.w4,
            self.b4,
        ]

    def run(self, x):
        first_layer_out = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        second_layer_out = nn.ReLU(
            nn.AddBias(nn.Linear(first_layer_out, self.w2), self.b2)
        )
        third_layer_out = nn.ReLU(
            nn.AddBias(nn.Linear(second_layer_out, self.w3), self.b3)
        )
        return nn.AddBias(nn.Linear(third_layer_out, self.w4), self.b4)

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        batch_size = 100
        loss = float("inf")
        valid_acc = 0
        while valid_acc < 0.97:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)
            valid_acc = dataset.get_validation_accuracy()
            if valid_acc > 0.95:
                self.lr = 0.05


class LanguageIDModel(object):

    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.lr = 0.1
        self.init_w = nn.Parameter(self.num_chars, 512)
        self.init_b = nn.Parameter(1, 512)

        self.mid_x_w = nn.Parameter(self.num_chars, 512)
        self.mid_h_w = nn.Parameter(512, 512)
        self.mid_b = nn.Parameter(1, 512)

        self.out_w1 = nn.Parameter(512, 128)
        self.out_b1 = nn.Parameter(1, 128)
        self.out_w2 = nn.Parameter(128, len(self.languages))
        self.out_b2 = nn.Parameter(1, len(self.languages))

        self.params = [
            self.init_w,
            self.init_b,
            self.mid_x_w,
            self.mid_h_w,
            self.mid_b,
            self.out_w1,
            self.out_b1,
            self.out_w2,
            self.out_b2,
        ]

    def run(self, xs):
        h_i = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.init_w), self.init_b))
        for char in xs[1:]:
            h_i = nn.ReLU(
                nn.AddBias(
                    nn.Add(nn.Linear(char, self.mid_x_w), nn.Linear(h_i, self.mid_h_w)),
                    self.mid_b,
                )
            )
        out1 = nn.ReLU(nn.AddBias(nn.Linear(h_i, self.out_w1), self.out_b1))
        return nn.AddBias(nn.Linear(out1, self.out_w2), self.out_b2)

    def get_loss(self, xs, y):
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        batch_size = 100
        loss = float("inf")
        valid_acc = 0

        while valid_acc < 0.85:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)
            valid_acc = dataset.get_validation_accuracy()
            if valid_acc > 0.8:
                self.lr = 0.01
