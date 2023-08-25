import math
import random

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

    def static_dots(self, grid):
        mu_1 = 2
        sigma_1 = 0.8
        mu_2 = 6
        sigma_2 = 0.4
        mu_3 = 4
        sigma_3 = 0.6
        mu_4 = 3
        sigma_4 = 1.2
        gaussian_1_x = np.random.normal(mu_1, sigma_1, 50)
        gaussian_1_y = np.random.normal(mu_2, sigma_2, 50)
        gaussian_2_x = np.random.normal(mu_3, sigma_3, 30)
        gaussian_2_y = np.random.normal(mu_4, sigma_4, 30)
        dummy_1 = np.zeros(50)
        dummy_2 = np.zeros(30)
        dots_1 = [Dot(grid.c2p(*point), color=YELLOW) for point in
                  list(map(lambda x, y, z: (x, y, z), gaussian_1_x, gaussian_1_y, dummy_1))]
        dots_2 = [Dot(grid.c2p(*point), color=GREEN) for point in
                  list(map(lambda x, y, z: (x, y, z), gaussian_2_x, gaussian_2_y, dummy_2))]
        curve_1 = grid.plot(lambda x: self.PDF_normal(x, mu_1, sigma_1), color=YELLOW)
        curve_2 = grid.plot(lambda x: self.PDF_normal(x, mu_3, sigma_3), color=GREEN)
        group_curve = VGroup(curve_1, curve_2)
        return VGroup(*dots_1, *dots_2), group_curve

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

        mu_1 = ValueTracker(3.5)
        sigma_1 = ValueTracker(0.5)
        mu_2 = ValueTracker(3)
        sigma_2 = ValueTracker(0.7)
        mu_3 = ValueTracker(6)
        sigma_3 = ValueTracker(0.9)
        mu_4 = ValueTracker(7)
        sigma_4 = ValueTracker(0.7)

        updating_dots = VGroup()
        dots_g1 = always_redraw(
            lambda: VGroup(*[Dot(point=grid.c2p(np.random.normal(mu_1.get_value(), sigma_1.get_value()),
                                       np.random.normal(mu_2.get_value(), sigma_2.get_value()),
                                       0), color=YELLOW) for _ in range(50)])
        )
        dots_g2 = always_redraw(
            lambda: VGroup(*[Dot(point=grid.c2p(np.random.normal(mu_3.get_value(), sigma_3.get_value()),
                                                np.random.normal(mu_4.get_value(), sigma_4.get_value()),
                                                0), color=GREEN) for _ in range(30)])
        )
        updating_dots.add(dots_g1, dots_g2)

        title = Title(
            "Mixture of two Gaussians",
            include_underline=False,
            font_size=40,
        ).shift(LEFT)
        self.play(FadeIn(title, grid, grid_labels))

        # Text to display distrubtion mean
        formula = MathTex(r"\mathcal{X} \sim \mathcal{N}({{\mu}},{{\sigma}}^{2})").to_edge(RIGHT + UP)

        g1_x_mu_text = MathTex(r'\mu_x^1 = ').next_to(formula, DOWN + LEFT, buff=0.2).set_color(YELLOW)
        g1_x_sigma_text = MathTex(r'\sigma_x^1 = ').next_to(g1_x_mu_text, RIGHT, buff=1.4).set_color(YELLOW)
        g2_x_mu_text = MathTex(r'\mu_x^2 = ').next_to(g1_x_mu_text, DOWN, buff=0.2).set_color(GREEN)
        g2_x_sigma_text = MathTex(r'\sigma_x^2 = ').next_to(g2_x_mu_text, RIGHT, buff=1.4).set_color(GREEN)

        # Always redraw the decimal value for mu for each frame
        g1_x_mu_value_text = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(mu_1.get_value())
                .next_to(g1_x_mu_text, RIGHT, buff=0.2)
                .set_color(YELLOW)
        )
        g1_x_sigma_value_text = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(sigma_1.get_value())
                .next_to(g1_x_sigma_text, RIGHT, buff=0.2)
                .set_color(YELLOW)
        )
        g2_x_mu_value_text = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(mu_3.get_value())
                .next_to(g2_x_mu_text, RIGHT, buff=0.2)
                .set_color(GREEN)
        )
        g2_x_sigma_value_text = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(sigma_3.get_value())
                .next_to(g2_x_sigma_text, RIGHT, buff=0.2)
                .set_color(GREEN)
        )

        g1_x = always_redraw(
            lambda: grid.plot(
                lambda x: self.PDF_normal(x, mu_1.get_value(), sigma_1.get_value()), color=YELLOW)
        )
        g2_x = always_redraw(
            lambda: grid.plot(
                lambda x: self.PDF_normal(x, mu_3.get_value(), sigma_3.get_value()), color=GREEN)
        )
        group_curves_1 = VGroup(g1_x, g2_x)
        group_text_1 = VGroup(g1_x_mu_text, g1_x_sigma_text, g2_x_mu_text, g2_x_sigma_text, g1_x_mu_value_text, g1_x_sigma_value_text, g2_x_mu_value_text, g2_x_sigma_value_text, formula)
        self.add(group_curves_1, group_text_1)
        self.play(Create(updating_dots))

        for _ in range(10):
            self.play(mu_1.animate.set_value(random.uniform(-2,8)),
                      mu_2.animate.set_value(random.uniform(-1,6)),
                      mu_3.animate.set_value(random.uniform(-1,7)),
                      mu_4.animate.set_value(random.uniform(1,7)),
                      sigma_1.animate.set_value(random.uniform(0,2)),
                      sigma_2.animate.set_value(random.uniform(0,2)),
                      sigma_3.animate.set_value(random.uniform(0,3)),
                      sigma_4.animate.set_value(random.uniform(0,2)),
                      run_time=10)

        self.wait(2)
        self.play(FadeOut(updating_dots))
        del updating_dots

        static_dots, group_curve_2 = self.static_dots(grid)
        initial_curve_1 = grid.plot(lambda x: self.PDF_normal(x, 1, 0.2), color=YELLOW)
        initial_curve_2 = grid.plot(lambda x: self.PDF_normal(x, 2, 0.3), color=GREEN)
        initial_curve_group = VGroup(initial_curve_1, initial_curve_2)
        self.play(FadeIn(static_dots),
                  ReplacementTransform(group_curves_1, initial_curve_group))
        self.play(*[dot.animate.set_color(WHITE) for dot in static_dots])
        responsibility = MathTex(
            r"r_{nk} = \frac{\pi_k\mathcal{N}(x_n|\mu_k, \Sigma_k))}{\sum_{j=1}^K\pi_j\mathcal{N}(x_n|\mu_j, \Sigma_j)}").shift(4*RIGHT + 2*UP).scale(0.8)
        new_title = Title("GMMs visual explanation (K=2)", include_underline=False, font_size=40).shift(LEFT)
        self.play(ReplacementTransform(group_text_1, responsibility),
                  ReplacementTransform(title, new_title))

        self.wait()

        # GMMs math
        indices = ValueTracker(0)
        dot = always_redraw(
            lambda: static_dots[int(indices.get_value())].set_color(BLUE))

        # selected point
        selected_point_text = Tex(r'Selected point:', color=BLUE).shift(4*RIGHT + UP).scale(0.8)
        selected_point_value_text_1 = always_redraw(
                    lambda: DecimalNumber(num_decimal_places=2)
                        .set_value(grid.p2c([dot.get_x(), dot.get_y(), 0])[0])
                        .next_to(selected_point_text, DOWN, buff=0.2)
                        .set_color(BLUE)
                        .scale(0.8)
                )
        selected_point_value_text_2 = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(grid.p2c([dot.get_x(), dot.get_y(), 0])[1])
                .next_to(selected_point_value_text_1, RIGHT, buff=0.2)
                .set_color(BLUE)
                .scale(0.8)
        )

        # intersecting point
        lines_1 = always_redraw(lambda: grid.get_lines_to_point((dot.get_x(), dot.get_y(), 0)))
        intersecting_point_text = Tex(r'Intersecting point:').shift(4*RIGHT).scale(0.8)

        intersecting_point_value_text_1 = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[0]))[0])
                .next_to(intersecting_point_text, DOWN, buff=0.2)
                .set_color(YELLOW)
                .scale(0.8)
        )
        intersecting_point_value_text_2 = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[0]))[1])
                .next_to(intersecting_point_value_text_1, RIGHT, buff=0.2)
                .set_color(YELLOW)
                .scale(0.8)
        )
        intersecting_point_value_text_3 = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[1]))[0])
                .next_to(intersecting_point_text, 4*DOWN, buff=0.2)
                .set_color(GREEN)
                .scale(0.8)
        )
        intersecting_point_value_text_4 = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[1]))[1])
                .next_to(intersecting_point_value_text_3, RIGHT, buff=0.2)
                .set_color(GREEN)
                .scale(0.8)
        )
        group_text_2 = VGroup(selected_point_text, selected_point_value_text_1, selected_point_value_text_2,
                              intersecting_point_text, intersecting_point_value_text_1, intersecting_point_value_text_2,
                              intersecting_point_value_text_3, intersecting_point_value_text_4)

        self.play(ShowCreationThenFadeOut(lines_1))
        self.add(group_text_2)
        self.wait()

        # responsibility display
        responsibility_text = MathTex(r"r_1 =", color=BLUE).shift(3 * RIGHT + 3*DOWN).scale(0.8)

        # denominator
        intersecting_point_value_text_2_target_copy = intersecting_point_value_text_2.copy()
        intersecting_point_value_text_2_target_copy.generate_target()
        intersecting_point_value_text_2_target_copy.target.next_to(responsibility_text, RIGHT + DOWN)
        intersecting_point_value_text_4.generate_target()
        intersecting_point_value_text_4.target.next_to(intersecting_point_value_text_2_target_copy.target, RIGHT, buff=0.5)

        intersecting_point_value_text_2_copy_b = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(
                grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[0]))[1])
                .next_to(responsibility_text, RIGHT+DOWN, buff=0)
                .set_color(YELLOW)
                .scale(0.8)
        )
        pi_text_1_b = MathTex(r"\pi_1+", color=YELLOW).next_to(intersecting_point_value_text_2_copy_b, RIGHT, buff=0)
        intersecting_point_value_text_4_copy = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(
                grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[1]))[1])
                .next_to(pi_text_1_b, RIGHT)
                .set_color(GREEN)
                .scale(0.8)
        )
        pi_text_2 = MathTex(r"\pi_2", color=GREEN).next_to(intersecting_point_value_text_4_copy, RIGHT, buff=0)

        denominator_group = VGroup(intersecting_point_value_text_2_copy_b, pi_text_1_b,
                                   pi_text_2, intersecting_point_value_text_4_copy)

        # line
        line = Line(LEFT, RIGHT)
        line.set_width(denominator_group.get_width() + 0.2)
        line.next_to(denominator_group, UP, buff=0.1, aligned_edge=LEFT)

        # numerator
        intersecting_point_value_text_2.generate_target()
        intersecting_point_value_text_2.target.next_to(line, UP, buff=0)

        intersecting_point_value_text_2_copy = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2)
                .set_value(
                grid.p2c(grid.input_to_graph_point(grid.p2c([dot.get_x(), dot.get_y(), 0])[0], initial_curve_group[0]))[1])
                .next_to(line, UP)
                .set_color(YELLOW)
                .scale(0.8)
        )
        pi_text_1 = MathTex(r"\pi_1", color=YELLOW).next_to(intersecting_point_value_text_2_copy, RIGHT, buff=0)
        numerator_group = VGroup(pi_text_1, intersecting_point_value_text_2_copy)

        y1 = intersecting_point_value_text_2.copy()
        y2 = intersecting_point_value_text_2_target_copy.copy()
        y3 = intersecting_point_value_text_4.copy()
        self.play(MoveToTarget(y1), MoveToTarget(y2), MoveToTarget(y3))
        self.play(FadeOut(y1),
                  FadeOut(y2),
                  FadeOut(y3))
        text_responsibility_update = VGroup(responsibility_text, numerator_group, line, denominator_group)
        self.add(text_responsibility_update)
        self.play(indices.animate.set_value(int(len(static_dots)) - 1), run_time=5)
        self.play(*[dot.animate.set_color(BLUE) for dot in static_dots])
        self.wait(2)

        # E step
        title_e_step = Title("EM Algorithm", include_underline=False, font_size=40).shift(LEFT)

        self.play(FadeOut(group_text_2),
                  FadeOut(responsibility),
                  FadeOut(text_responsibility_update),
                  ReplacementTransform(new_title, title_e_step))
        tex_initialize = Tex(r"1. Initialize $\pi$, $\mu$, $\sigma$").shift(4*RIGHT + 2*UP).scale(0.7)
        tex_e_step = Tex(r"2. Compute $r$").next_to(tex_initialize, DOWN).scale(0.7)
        tex_m_step = Tex(r"3. Update  $\pi$, $\mu$, $\sigma$ given the new $r$").next_to(tex_e_step, DOWN).scale(0.7)
        tex_repeat = Tex(r"4. Repeat until convergence.").next_to(tex_m_step, DOWN).scale(0.7)
        self.play(Create(tex_initialize),
                  Create(tex_e_step),
                  Create(tex_m_step),
                  Create(tex_repeat)
                  )
        self.wait(2)
        initial_curve_1b = grid.plot(lambda x: self.PDF_normal(x, 1.5, 0.4), color=YELLOW)
        initial_curve_2b = grid.plot(lambda x: self.PDF_normal(x, 3, 0.2), color=GREEN)
        initial_curve_groupb = VGroup(initial_curve_1b, initial_curve_2b)
        self.play(ReplacementTransform(initial_curve_group, initial_curve_groupb))
        self.wait(2)
        self.play(ReplacementTransform(initial_curve_groupb, group_curve_2))
        self.wait(2)


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
