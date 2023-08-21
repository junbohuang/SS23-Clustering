from manim import *


class ProblemRepresentation(Scene):
    def construct(self):
        # data
        german_cities_data = {"Hamburg": ["53.55", "9.99"], "Bremen": ["53.07", "8.80"],
                              "Hanover": ["52.37", "9.73"], "Leipzig":  ["51.33", "12.37"],
                              "Frankfurt": ["50.13", "8.66"], "Nürnberg": ["49.45", "11.07"]}

        # 1. Introduction to the problem and representation we want - view table as dataset
        table_1 = Table(list(german_cities_data.values()),
                        row_labels=[Text(city, color=BLUE) for city in list(german_cities_data.keys())],
                        col_labels=[Text("Latitude", color=RED), Text("Longtitude", color=RED)],
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
        text_2_2 = Text(", where there are rows (individuals) and columns (features)",
                        t2c={"rows (individuals)": BLUE,
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
        tex_5_1 = MathTex(r'd(x_i,x_j) = geodesic(x_1, x_2)', font_size=38, color=YELLOW).next_to(tex_4_2, 4*LEFT)
        self.play(Write(tex_5_1))
        self.wait(2)

        # tex_5_2 = MathTex(r'd(x_i,x_j) &= \text{geodesic(x_1, x_2)} \\ &= euclidean(x_1, x_2) \\ &= cosine(x_1, x_2)', font_size=38)

        # 6. sentence embedding
        matrix_6 = Matrix([[0.28, 0.58, "...", 0.88], [0.11, 0.17, "...", 0.45],
                           [0.45, 0.22, "...", 0.18], [0.35, 0.33, "...", 0.51],
                           [0.26, 0.66, "...", 0.73], [0.05, 0.74, "...", 0.11],
                           [0.54, 0.52, "...", 0.21]]).scale(0.7)
        # self.play(FadeTransform(matrix_3, matrix_6))
        tex_6_1 = MathTex(r'd(x_i,x_j) = cosine(x_1, x_2)', font_size=42, color=YELLOW).next_to(tex_4_2, 6*LEFT + UP)
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
        text_8_1 = Text("Curse of dimensionality \n\n "
                        "the more features you have, \n"
                        "the more data your need.",
                        t2c={"features": BLUE,
                             "data": RED},
                        font_size=50)
        self.play(Write(text_8_1, run_time=2))
        self.wait()

