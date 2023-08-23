import math

from manim import *
import numpy as np
from scipy.stats import norm


class ProblemRepresentation(Scene):
    def construct(self):
        # data
        german_cities_data = {"Hamburg": ["53.55", "9.99"], "Bremen": ["53.07", "8.80"],
                              "Hanover": ["52.37", "9.73"], "Leipzig":  ["51.33", "12.37"],
                              "Frankfurt": ["50.13", "8.66"], "Nürnberg": ["49.45", "11.07"]}

        # 1. Introduction to the problem and representation we want - view table as dataset
        table_1 = Table(list(german_cities_data.values()),
                        row_labels=[Text(city, color=BLUE) for city in list(german_cities_data.keys())],
                        col_labels=[Text("Latitude", color=RED), Text("Longitude", color=RED)],
                        v_buff=0.8).scale(0.6).shift(UP)
        text_1 = Text("Given a table of some cities, with their latitude and longitude.",
                        t2c={"cities": BLUE,
                             "latitude": RED,
                             "longitude": RED},
                      font_size=34).next_to(table_1, DOWN, buff=0.3)

        self.play(Create(table_1, run_time=2),
                  Create(text_1), run_time=2)
        self.wait(2)

        # 2. we can view this table as our dataset, where we have rows, or individuals, and columns, or features.
        text_2_1 = Tex(r"We can call this table the dataset $\mathtt{x}$").next_to(table_1, DOWN, buff=0.3)
        text_2_2 = Text(", where there are rows (observations) and columns (features)",
                        t2c={"rows (observations)": BLUE,
                             "columns (features)": RED},
                        font_size=34).next_to(text_2_1, DOWN, buff=0.1)
        text_2_3 = Tex(r'$\mathtt{x} =$').next_to(table_1, LEFT)
        text_group_2 = VGroup(text_2_1, text_2_2)
        box_2_1 = SurroundingRectangle(table_1.get_rows()[1], BLUE)
        box_2_2 = SurroundingRectangle(table_1.get_columns()[1], RED)
        self.play(
            ReplacementTransform(text_1, text_group_2, run_time=2),
            Create(text_2_3, run_time=2),
            Create(box_2_1, run_time=2),
            Create(box_2_2, run_time=2)
        )
        self.wait(2)

        # 3. Matrix representation of the dataset
        matrix_3 = Matrix(list(german_cities_data.values()))
        text_3 = Text("Let's represent our dataset as a n x m matrix",
                      t2c={"n ": RED,
                           "m ": BLUE},
                      font_size=34).next_to(matrix_3, DOWN)
        text_3_3 = Tex(r'$\mathtt{x} =$').next_to(matrix_3, LEFT)
        box_3_1 = SurroundingRectangle(matrix_3.get_rows()[1], BLUE)
        box_3_2 = SurroundingRectangle(matrix_3.get_columns()[1], RED)

        self.play(
            ReplacementTransform(table_1, matrix_3, run_time=2),
            ReplacementTransform(text_group_2, text_3, run_time=2),
            FadeOut(text_1),
            FadeOut(text_2_3),
            FadeOut(box_2_1),
            FadeOut(box_2_2),
            FadeIn(box_3_1),
            FadeIn(box_3_2),
            FadeIn(text_3_3),
            run_time=2)

        self.wait(2)

        # 4. This dataset is a city coordination dataset
        tex_4_1 = VGroup(*[Tex(fr"$\leftarrow$ {city}", font_size=40, color=RED)
                      .next_to(matrix_3.get_rows()[i], RIGHT, buff=0.5)
                      for i, city in enumerate(list(german_cities_data.keys()))
                       ]
                      )

        text_4_1 = Text("Latitude", font_size=25, color=BLUE).next_to(matrix_3.get_columns()[0], 4*UP)
        text_4_2 = Text("Longitude", font_size=25, color=BLUE).next_to(matrix_3.get_columns()[0], 4*UP + 3*RIGHT)
        tex_4_2 = Tex(r"$\downarrow$").next_to(matrix_3.get_columns()[0], UP)
        tex_4_3 = Tex(r"$\downarrow$").next_to(matrix_3.get_columns()[1], UP)
        group_4_1 = VGroup(text_4_1, text_4_2, tex_4_2, tex_4_3)

        self.play(Create(tex_4_1), Create(group_4_1))
        self.wait(2)

        # 5. distance function
        tex_5_1 = MathTex(r'd(x_i,x_j) = geodesic(x_i, x_j)', font_size=38, color=YELLOW).next_to(tex_4_2, 4*LEFT)
        self.play(Write(tex_5_1))
        self.wait(2)

        # tex_5_2 = MathTex(r'd(x_i,x_j) &= \text{geodesic(x_1, x_2)} \\ &= euclidean(x_1, x_2) \\ &= cosine(x_1, x_2)', font_size=38)

        # 6. sentence embedding
        matrix_6 = Matrix([[0.28, 0.58, "...", 0.88], [0.11, 0.17, "...", 0.45],
                           [0.45, 0.22, "...", 0.18], [0.35, 0.33, "...", 0.51],
                           [0.26, 0.66, "...", 0.73], [0.05, 0.74, "...", 0.11],
                           [0.54, 0.52, "...", 0.21]]).scale(0.7)
        # self.play(FadeTransform(matrix_3, matrix_6))
        tex_6_1 = MathTex(r'd(x_i,x_j) = cosine(x_i, x_j)', font_size=42, color=YELLOW).next_to(tex_4_2, 6*LEFT + UP)
        text_6_1 = MathTex(r"f_1", font_size=42, color=BLUE).next_to(matrix_6.get_columns()[0], 4*UP)
        text_6_2 = MathTex(r"f_2", font_size=42, color=BLUE).next_to(matrix_6.get_columns()[1], 4*UP)
        text_6_3 = MathTex(r"f_m", font_size=42, color=BLUE).next_to(matrix_6.get_columns()[3], 4*UP)
        tex_6_11 = Tex(r"$\downarrow$", color=BLUE).next_to(matrix_6.get_columns()[0], UP)
        tex_6_12 = Tex(r"$\downarrow$", color=BLUE).next_to(matrix_6.get_columns()[1], UP)
        tex_6_13 = Tex(r"$\downarrow$", color=BLUE).next_to(matrix_6.get_columns()[3], UP)
        tex_6_4 = Tex(r"  $\leftarrow$This is a cat.", font_size=38, color=RED).next_to(matrix_6.get_rows()[0], RIGHT)
        tex_6_5 = Tex(r"  $\leftarrow$This is a dog.", font_size=38, color=RED).next_to(matrix_6.get_rows()[1], RIGHT)
        tex_6_6 = Tex(r"  $\leftarrow$I am sad.", font_size=38, color=RED).next_to(matrix_6.get_rows()[2], RIGHT)
        tex_6_7 = Tex(r"  $\leftarrow$Home alone.", font_size=38, color=RED).next_to(matrix_6.get_rows()[3], RIGHT)
        tex_6_8 = Tex(r"  $\leftarrow$Data engineering", font_size=38, color=RED).next_to(matrix_6.get_rows()[4], RIGHT)
        tex_6_9 = Tex(r"  $\leftarrow$Amazing, man", font_size=38, color=RED).next_to(matrix_6.get_rows()[5], RIGHT)
        tex_6_10 = Tex(r"  $\leftarrow$But no mensa...", font_size=38, color=RED).next_to(matrix_6.get_rows()[6], RIGHT)
        text_6_14 = Text("We can change a dataset with high-dimensional features.",
                         t2c={"features": BLUE},
                         font_size=32).next_to(matrix_6, DOWN)
        text_6_15 = Tex(r'$\mathtt{x} =$', font_size=40).next_to(matrix_6, LEFT)

        group_6 = VGroup(tex_6_4, tex_6_5, tex_6_6, tex_6_7, tex_6_8, tex_6_9, tex_6_10)
        group_7 = VGroup(text_6_1, text_6_2, text_6_3, tex_6_11, tex_6_12, tex_6_13)
        # self.play(FadeOut(text_group_2), FadeOut(text_1), FadeOut(table_1))
        self.play(
                  ReplacementTransform(matrix_3, matrix_6),
                  ReplacementTransform(tex_4_1, group_6),
                  ReplacementTransform(text_3, text_6_14),
                  ReplacementTransform(group_4_1, group_7),
                  ReplacementTransform(tex_5_1, tex_6_1),
                  ReplacementTransform(text_3_3, text_6_15),
                  FadeOut(box_3_1),
                  FadeOut(box_3_2),
                  run_time=2)
        self.wait(2)

        # 7. distance in high-dimensional space
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )
        text_7_1 = Text("Distance in 2-dimensional space ",
                        t2c={"2": BLUE},
                        font_size=50)
        self.play(Write(text_7_1))
        self.play(text_7_1.animate.shift(2*UP).scale(0.4))
        self.wait()

        text_7_2 = MathTex(r"d(x_{{i}}, x_{{j}})= x_{{{1}}{{i}}}\cdot x_{{{1}}{{j}}} "
                           r"+ x_{{{2}}{{i}}}\cdot x_{{{2}}{{j}}}",
                           font_size=50)

        text_7_2[1].set_color(RED)
        text_7_2[3].set_color(RED)
        text_7_2[6].set_color(RED)
        text_7_2[9].set_color(RED)
        text_7_2[12].set_color(RED)
        text_7_2[15].set_color(RED)
        text_7_2[5].set_color(BLUE)
        text_7_2[8].set_color(BLUE)
        text_7_2[11].set_color(BLUE)
        text_7_2[14].set_color(BLUE)

        self.play(Write(text_7_2))
        self.play(text_7_2.animate.shift(UP).scale(0.9))
        self.wait()

        text_7_3 = Text("Distance in m-dimensional space ",
                        t2c={" m": BLUE},
                        font_size=50)
        self.play(Write(text_7_3))
        self.play(text_7_3.animate.shift(2*DOWN).scale(0.4))
        self.wait()

        text_7_4 = MathTex(r"d(x_{{i}}, x_{{j}})=x_{{{1}}{{i}}}\cdot x_{{{1}}{{j}}}"
                           r"+x_{{{2}}{{i}}}\cdot x_{{{2}}{{j}}} + ... +  x_{{{m}}{{i}}}"
                           r"\cdot x_{{{m}}{{j}}}",
                           font_size=50)
        text_7_4[1].set_color(RED)
        text_7_4[3].set_color(RED)
        text_7_4[6].set_color(RED)
        text_7_4[9].set_color(RED)
        text_7_4[12].set_color(RED)
        text_7_4[15].set_color(RED)
        text_7_4[18].set_color(RED)
        text_7_4[21].set_color(RED)

        text_7_4[5].set_color(BLUE)
        text_7_4[8].set_color(BLUE)
        text_7_4[11].set_color(BLUE)
        text_7_4[14].set_color(BLUE)
        text_7_4[17].set_color(BLUE)
        text_7_4[20].set_color(BLUE)

        self.play(Write(text_7_4))
        self.play(text_7_4.animate.shift(DOWN).scale(0.9))
        self.wait()

        # 8. curse of dimensionality
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )
        text_8_1 = Text("Curse of dimensionality \n\n"
                        "The more features you have, \n"
                        "The more data your need to fill up the space.",
                        t2c={"features": BLUE,
                             "data": RED},
                        font_size=50)
        self.play(Write(text_8_1, run_time=2))
        self.wait(2)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )

        # vis
        line = NumberLine(
            x_range=[-10, 10, 2],
            length=13,
            include_numbers=True,
        )
        dots_1d = [Dot([n - 3, 0, 0], color=YELLOW) for n in np.random.sample(50) * 10] + [
            Dot([n + 3, 0, 0], color=YELLOW) for n in np.random.sample(50) * -10]
        self.play(FadeIn(line))
        self.play(Create(VGroup(*dots_1d)))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )

        plane = NumberPlane(
            x_range=[-10, 10, 2],
            y_range=[-10, 10, 2],
            x_length=13,
            y_length=10
        )
        dots_2d = [Dot([point[0], point[1], 0], color=YELLOW) for point in
                   np.random.uniform(low=-5, high=5, size=(100, 2))]
        self.play(FadeIn(plane))
        self.play(Create(VGroup(*dots_2d)))
        self.wait(2)


class GMMs(Scene):
    def PDF_normal(self, x, mu, sigma):
        '''
        General form of probability density function of univariate normal distribution
        '''
        return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

    def construct(self):
        grid = Axes(
            x_range=[-2, 10, 2],  # step size determines num_decimal_places.
            y_range=[-2, 10, 2],
            x_length=8,
            y_length=6,
            axis_config={
                "numbers_to_include": np.arange(-4, 11, 2),
                "font_size": 24,
            },
            tips=False,
        ).shift(0.4*DOWN + 2*LEFT)

        # Labels for the x-axis and y-axis.
        y_label = grid.get_y_axis_label("y")
        x_label = grid.get_x_axis_label("x")
        grid_labels = VGroup(x_label, y_label)

        mu_1 = 3.5
        sigma_1 = 0.5
        mu_2 = 3
        sigma_2 = 0.7
        mu_3 = 6
        sigma_3 = 0.9
        mu_4 = 7
        sigma_4 = 0.7

        gaussian_1_x = np.random.normal(mu_1, sigma_1, 30)
        gaussian_1_y = np.random.normal(mu_2, sigma_2, 30)
        gaussian_2_x = np.random.normal(mu_3, sigma_3, 50)
        gaussian_2_y = np.random.normal(mu_4, sigma_4, 50)
        dummy_1 = np.zeros(30)
        dummy_2 = np.zeros(50)

        dots_1 = [Dot(grid.c2p(*point), color=YELLOW) for point in list(map(lambda x, y, z:(x, y, z), gaussian_1_x, gaussian_1_y, dummy_1))]
        dots_2 = [Dot(grid.c2p(*point), color=GREEN) for point in list(map(lambda x, y, z:(x, y, z), gaussian_2_x, gaussian_2_y, dummy_2))]

        graphs = VGroup(*dots_1, *dots_2)
        title = Title(
            # spaces between braces to prevent SyntaxError
            "Mixture of two Gaussians",
            include_underline=False,
            font_size=40,
        ).shift(LEFT)
        self.play(FadeIn(title, graphs, grid, grid_labels))

        self.wait()

        # add text
        text_group = VGroup()
        formula = MathTex(r"\mathcal{X} \sim \mathcal{N}({{\mu}},{{\sigma}}^{2})").next_to(grid, RIGHT + UP)
        formula[1].set_color(MAROON)  # mu
        formula[3].set_color(PURPLE)  # sigma
        text_group += formula

        # first gaussian
        tex_1 = MathTex(r"\mu_1")
        tex_5 = MathTex(r"\sigma_1")
        text_group += tex_1
        text_group += tex_5
        # second gaussian
        tex_2 = MathTex(r"\mu_2")
        tex_6 = MathTex(r"\sigma_2")
        text_group += tex_2
        text_group += tex_6
        # third gaussian
        tex_3 = MathTex(r"\mu_3")
        tex_7 = MathTex(r"\sigma_3")
        text_group += tex_3
        text_group += tex_7
        # forth gaussian
        tex_4 = MathTex(r"\mu_4")
        tex_8 = MathTex(r"\sigma_4")
        text_group += tex_4
        text_group += tex_8

        # create variables
        var_1 = Variable(float(mu_1), tex_1, num_decimal_places=3).next_to(formula, DOWN).scale(0.6)
        var_1.value.set_color(MAROON)  # g1 x mu
        var_1.label.set_color(YELLOW)  # g1 x mu
        var_2 = Variable(float(mu_2), tex_2, num_decimal_places=3).next_to(var_1, DOWN).scale(0.6)
        var_2.value.set_color(MAROON)  # g1 y mu
        var_2.label.set_color(YELLOW)  # g1 y mu
        var_3 = Variable(float(mu_3), tex_3, num_decimal_places=3).next_to(var_2, DOWN).scale(0.6)
        var_3.value.set_color(MAROON)  # g2 x mu
        var_3.label.set_color(GREEN)  # g2 x mu
        var_4 = Variable(float(mu_4), tex_4, num_decimal_places=3).next_to(var_3, DOWN).scale(0.6)
        var_4.value.set_color(MAROON)  # g2 y mu
        var_4.label.set_color(GREEN)  # g2 y mu

        var_5 = Variable(float(sigma_1), tex_5, num_decimal_places=3).next_to(var_1, RIGHT).scale(0.6)
        var_5.value.set_color(PURPLE)  # g1 x sigma
        var_5.label.set_color(YELLOW)  # g1 x sigma
        var_6 = Variable(float(sigma_2), tex_6, num_decimal_places=3).next_to(var_2, RIGHT).scale(0.6)
        var_6.value.set_color(PURPLE)  # g1 y sigma
        var_6.label.set_color(YELLOW)  # g1 y sigma
        var_7 = Variable(float(sigma_3), tex_7, num_decimal_places=3).next_to(var_3, RIGHT).scale(0.6)
        var_7.value.set_color(PURPLE)  # g2 x sigma
        var_7.label.set_color(GREEN)  # g2 x sigma
        var_8 = Variable(float(sigma_4), tex_8, num_decimal_places=3).next_to(var_4, RIGHT).scale(0.6)
        var_8.value.set_color(PURPLE)  # g2 y sigma
        var_8.label.set_color(GREEN)  # g2 y sigma

        text_group += var_1
        text_group += var_2
        text_group += var_3
        text_group += var_4
        text_group += var_5
        text_group += var_6
        text_group += var_7
        text_group += var_8

        self.add(text_group)
        self.wait(0.5)

        # add Gaussian
        gaussian_curve_1 = always_redraw(
            lambda: grid.plot(
                lambda x: self.PDF_normal(x, var_1.value.get_value(), var_5.value.get_value()), color=YELLOW)
        )

        gaussian_curve_3 = always_redraw(
            lambda: grid.plot(
                lambda x: self.PDF_normal(x, var_3.value.get_value(), var_7.value.get_value()), color=GREEN)
        )

        self.play(Create(gaussian_curve_1),
                  Create(gaussian_curve_3))
        self.wait()

        # move dots
        gaussian_1_x_updated = np.random.normal(mu_1+1.500, sigma_1+0.3, 30)
        gaussian_1_y_updated = np.random.normal(mu_2-1.200, sigma_2+1, 30)
        gaussian_2_x_updated = np.random.normal(mu_3-1.500, sigma_3+0.5, 30)
        gaussian_2_y_updated = np.random.normal(mu_4+1.300, sigma_4+0.3, 30)

        dots_updated_1 = [Dot(grid.c2p(*point), color=YELLOW) for point in
                          list(map(lambda x, y, z:(x, y, z), gaussian_1_x_updated, gaussian_1_y_updated, dummy_1))]
        dots_updated_2 = [Dot(grid.c2p(*point), color=GREEN) for point in
                          list(map(lambda x, y, z: (x, y, z), gaussian_2_x_updated, gaussian_2_y_updated, dummy_1))]

        self.play(var_1.tracker.animate.set_value(mu_1+1.500),
                  var_2.tracker.animate.set_value(mu_2-1.500),
                  var_3.tracker.animate.set_value(mu_3-1.200),
                  var_4.tracker.animate.set_value(mu_4+1.300),
                  var_5.tracker.animate.set_value(sigma_1 + 0.3),
                  var_6.tracker.animate.set_value(sigma_2 + 1),
                  var_7.tracker.animate.set_value(sigma_3 + 0.5),
                  var_8.tracker.animate.set_value(sigma_4 + 0.300),
                  ReplacementTransform(VGroup(*dots_1), VGroup(*dots_updated_1)),
                  ReplacementTransform(VGroup(*dots_2), VGroup(*dots_updated_2)))

        self.wait(0.5)

        self.play(
            *[dot.animate.set_color(WHITE) for dot in graphs]
        )

        # GMM math
        self.play(var_1.tracker.animate.set_value(2),
                  var_2.tracker.animate.set_value(6),
                  var_3.tracker.animate.set_value(4),
                  var_4.tracker.animate.set_value(3),
                  var_5.tracker.animate.set_value(0.6),
                  var_6.tracker.animate.set_value(0.41),
                  var_7.tracker.animate.set_value(0.8),
                  var_8.tracker.animate.set_value(1.2))
        responsibility = MathTex(r"r_{nk} = \frac{\pi_k\mathcal{N}(x_n|\mu_k, \Sigma_k))}{\sum_{j=1}^K\pi_j\mathcal{N}(x_n|\mu_j, \Sigma_j)}").next_to(grid, RIGHT+UP).scale(0.6)

        self.play(Uncreate(text_group),
                  Uncreate(title),
                  Write(responsibility))
        title = Title("GMMs visual explanation (K=2)", include_underline=False, font_size=40).to_edge(LEFT + UP)
        self.add(title)

        for i, dot in enumerate(graphs):
            self.play(dot.animate.set_color(BLUE),)
            points = grid.p2c([dot.get_x(), dot.get_y(), 0])
            selected_point = Tex(fr"Selected point: \newline {np.round(points, decimals=2)}", color=BLUE).next_to(formula, DOWN).scale(0.6)

            lines_1 = grid.get_lines_to_point((dot.get_x(), dot.get_y(), 0))
            self.play(Create(lines_1), Create(selected_point))
            # self.wait()
            # show point coord
            coords_1 = grid.p2c(grid.input_to_graph_point(dot.get_x(), gaussian_curve_1))
            coords_2 = grid.p2c(grid.input_to_graph_point(dot.get_x(), gaussian_curve_3))
            intersected_point = Tex(fr"Intersecting point: \newline ("
                                    fr"{np.round(coords_1, decimals=2)[0]}, "
                                    fr"{np.round(coords_1, decimals=2)[1]}), "
                                    fr"({np.round(coords_2, decimals=2)[0]}, "
                                    fr"{np.round(coords_2, decimals=2)[1]})", color=YELLOW).next_to(selected_point, DOWN).scale(0.6)
            self.play(Create(intersected_point))
            if i == 0:
                self.wait(2)
            else:
                continue
        # var_g_1 = Variable(grid.input_to_graph_point(graphs[0].get_x(), gaussian_curve_1), tex_1, num_decimal_places=3).next_to(formula, DOWN).scale(0.6)


class CurseOfDimensionality(Scene):
    def construct(self):
        line = NumberLine(
            x_range=[-10, 10, 2],
            length=13,
            include_numbers=True,
        )
        dots_1d = [Dot([n-5, 0, 0], color=YELLOW) for n in np.random.sample(50)*10] + [Dot([n-5, 0, 0], color=YELLOW) for n in np.random.sample(50)*-10]
        self.play(FadeIn(line))
        self.play(Create(VGroup(*dots_1d)))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

        plane = NumberPlane(
            x_range=[-10, 10, 2],
            y_range=[-10, 10, 2],
            x_length=13,
            y_length=10
        )
        dots_2d = [Dot([point[0], point[1], 0], color=YELLOW) for point in np.random.uniform(low=-5, high=5, size=(100,2))]
        self.play(FadeIn(plane))
        self.play(Create(VGroup(*dots_2d)))
        self.wait(2)
