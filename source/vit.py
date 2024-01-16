import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        embedding_dim,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding_layer = tf.keras.layers.Conv2D(
            filters=self.embedding_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
        )

        self.cls_token = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="position_embedding",
        )

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        input: (batch_size, image_size, image_size, channels) = (b, h, w, c)
        return: (batch_size, num_patches + 1, embedding_dim) = (b, n, d)

        """
        # (b, h, w, c) -> (b, h/p, w/p, d), where p is patch_size
        embed = self.patch_embedding_layer(input)

        # (b, h/p, w/p, d) -> (b, num_pathes, d)
        embed = tf.reshape(embed, [-1, self.num_patches, self.embedding_dim])

        # (b, num_pathes, d) -> (b, num_pathes + 1, d),
        # where cls_token is broadcasted to (b, 1, d)
        embed = tf.concat(
            [
                tf.broadcast_to(
                    self.cls_token, [tf.shape(embed)[0], 1, self.embedding_dim]
                ),
                embed,
            ],
            axis=1,
        )

        # (b, n, d) -> (b, n, d)
        embed = embed + self.position_embedding

        return embed


class MlpBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        dropout,
    ):
        super().__init__()
        self.dense_layer_1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dropout_layer_1 = tf.keras.layers.Dropout(dropout)
        self.dense_layer_2 = tf.keras.layers.Dense(embedding_dim)
        self.dropout_layer_2 = tf.keras.layers.Dropout(dropout)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        input: (batch_size, num_patches + 1, embedding_dim) = (b, n, d)
        return: (batch_size, num_patches + 1, embedding_dim) = (b, n, d)
        """
        output = self.dense_layer_1(input)
        output = self.dropout_layer_1(output)
        output = self.dense_layer_2(output)
        output = self.dropout_layer_2(output)
        return output


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout,
    ):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
        )
        self.mhsa_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
        )
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
        )
        self.mlp = MlpBlock(hidden_dim, embedding_dim, dropout)

    def call(self, input: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        input: (batch_size, num_patches + 1, embedding_dim) = (b, n, d)
        return:
            (batch_size, num_patches + 1, embedding_dim) = (b, n, d)
            (batch_size, num_heads, num_patches + 1, num_patches + 1) = (b, h, n, n),
            where h is num_heads
        """
        output = self.layer_norm_1(input)
        output, attention_scores = self.mhsa_layer(
            output, output, return_attention_scores=True
        )
        output = output + input

        output = self.mlp(self.layer_norm_2(output)) + output

        return output, attention_scores


class VisionTransformerClassifier(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size=32,
        patch_size=16,
        embedding_dim=64,
        num_blocks=8,
        num_heads=4,
        hidden_dim_rate=4,
        num_classes=1,
        dropout=0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.embedding_layer = EmbeddingLayer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
        )

        self.encoder_blocks = [
            EncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim_rate * embedding_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]

        self.mlp_head_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
        )
        if num_classes == 1:
            self.mlp_head_dense = tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            self.mlp_head_dense = tf.keras.layers.Dense(
                num_classes, activation="softmax"
            )

    def call(self, input: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        input: (batch_size, image_size, image_size, channels) = (b, h, w, c)
        return:
            (batch_size, num_classes) = (b, c),
            where c is num_classes
            (batch_size, num_blocks, num_heads, num_patches + 1, num_patches + 1) = (b, l, h, n, n),
            where l is num_blocks
        """
        # (b, h, w, c) -> (b, n, d)
        embed = self.embedding_layer(input)

        # (b, n, d) -> (b, n, d) and (num_blocks, b, h, n, n)
        attention_scores = []
        for encoder_block in self.encoder_blocks:
            embed, attention_score = encoder_block(embed)
            attention_scores.append(attention_score)

        # (b, n, d) -> (b, c)
        # output = self.mlp_head(embed[:, 0])
        output = self.mlp_head_dense(self.mlp_head_ln(embed[:, 0]))

        return output, tf.stack(attention_scores, axis=1)


class VisionTransformerVisualizer(tf.keras.layers.Layer):
    def __init__(self, vit_classifier):
        super().__init__()

        self.vit_classifier = vit_classifier

        image_size = vit_classifier.image_size
        patch_size = vit_classifier.patch_size

        self.size = image_size // patch_size
        self.upsampling_layer = tf.keras.layers.UpSampling2D(
            size=patch_size, interpolation="bilinear"
        )

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        input: (batch_size, image_size, image_size, channels) = (b, h, w, c)
        return:
            (batch_size, height, width) = (b, h, w)
        """

        # (b, h, w, c) -> (b, l, h, n, n)
        _, attention_scores = self.vit_classifier(input)

        # (b, l, h, n, n) -> (b, l, n, n)
        weights = tf.reduce_mean(attention_scores, axis=2)

        # (b, l, n, n) -> (b, l, n, n)
        weights = weights + tf.eye(tf.shape(weights)[2])

        # (b, l, n, n) -> (b, l, n, n)
        weights = weights / tf.reduce_sum(weights, axis=[2, 3], keepdims=True)

        # (b, l, n, n) -> (b, n, n)
        v = weights[:, -1, :, :]
        for n in range(1, tf.shape(weights)[1]):
            v = tf.matmul(v, weights[:, -(n + 1), :, :])

        # (b, n, n) -> (b, num_patches)
        weights = v[:, 0, 1:]

        # (b, num_patches) -> (b, h/p, w/p, 1), where num_patches = (h/p) * (w/p)
        weights = tf.reshape(
            weights,
            [-1, self.size, self.size, 1],
        )

        # (b, h/p, w/p, 1) -> (b, h, w, 1)
        weights = self.upsampling_layer(weights)

        # (b, h, w, 1) -> (b, h, w)
        weights = tf.squeeze(weights, axis=3)

        # (b, h, w) -> (b, h, w)
        attention_map = (
            weights - tf.reduce_min(weights, axis=[1, 2], keepdims=True)
        ) / (
            tf.reduce_max(weights, axis=[1, 2], keepdims=True)
            - tf.reduce_min(weights, axis=[1, 2], keepdims=True)
        )

        return attention_map

    def attention_rollout(self, input: tf.Tensor) -> tf.Tensor:
        """
        input: (batch_size, image_size, image_size, channels) = (b, h, w, c)
        return:
            (batch_size, height, width) = (b, h, w)
        """

        # (b, h, w, c) -> (b, l, h, n, n)
        _, attention_scores = self.vit_classifier(input)

        # (b, l, h, n, n) -> (b, l, n, n)
        weights = tf.reduce_mean(attention_scores, axis=2)

        # (b, l, n, n) -> (b, l, n, n)
        weights = weights + tf.eye(tf.shape(weights)[2])

        # (b, l, n, n) -> (b, l, n, n)
        weights = weights / tf.reduce_sum(weights, axis=[2, 3], keepdims=True)

        # (b, l, n, n) -> (b, n, n)
        v = weights[:, -1, :, :]
        for n in range(1, tf.shape(weights)[1]):
            v = tf.matmul(v, weights[:, -(n + 1), :, :])

        # (b, n, n) -> (b, num_patches)
        weights = v[:, 0, 1:]

        # (b, num_patches) -> (b, h/p, w/p, 1), where num_patches = (h/p) * (w/p)
        weights = tf.reshape(
            weights,
            [-1, self.size, self.size, 1],
        )

        return weights

    def upsample(self, weights: tf.Tensor) -> tf.Tensor:
        # (b, h/p, w/p, 1) -> (b, h, w, 1)
        weights = self.upsampling_layer(weights)
        return weights

    def normalize(self, weights: tf.Tensor) -> tf.Tensor:
        # (b, h, w, 1) -> (b, h, w)
        weights = tf.squeeze(weights, axis=3)

        # (b, h, w) -> (b, h, w)
        attention_map = (
            weights - tf.reduce_min(weights, axis=[1, 2], keepdims=True)
        ) / (
            tf.reduce_max(weights, axis=[1, 2], keepdims=True)
            - tf.reduce_min(weights, axis=[1, 2], keepdims=True)
        )

        return attention_map


def training_model(
    model,
    X_train,
    y_train,
    learning_rate=0.0001,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=True,
    )
    return model


def make_vit_classifier_model(
    image_size=64,
    patch_size=8,
    embedding_dim=64,
    num_blocks=8,
    num_heads=4,
    hidden_dim_rate=4,
    num_classes=1,
    dropout=0.1,
):
    vit_classifier = VisionTransformerClassifier(
        image_size=image_size,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_dim_rate=hidden_dim_rate,
        num_classes=num_classes,
        dropout=dropout,
    )
    input = tf.keras.Input(shape=(image_size, image_size, 1))
    output, _ = vit_classifier(input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def load_vit_visualizer(
    image_size=64,
    patch_size=8,
    embedding_dim=64,
    num_blocks=8,
    num_heads=4,
    hidden_dim_rate=4,
    num_classes=1,
    model_path="brain.h5",
):
    model = make_vit_classifier_model(
        image_size=image_size,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_dim_rate=hidden_dim_rate,
        num_classes=num_classes,
    )
    model.load_weights(model_path)
    vit_visualizer = VisionTransformerVisualizer(model.layers[1])
    return vit_visualizer
