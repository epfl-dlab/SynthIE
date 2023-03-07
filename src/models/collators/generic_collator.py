class GenericCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.params = kwargs

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_input(self, batch):
        if isinstance(batch, str):
            batch = [batch]

        collated_batch = {}

        max_length = self.params.get("max_input_length", None)
        tokenizer_output = self.tokenizer(
            batch,
            return_tensors="pt",
            return_attention_mask=True,
            padding=self.params["padding"],
            max_length=max_length,
            truncation=self.params["truncation"],
        )
        for k, v in tokenizer_output.items():
            prefix = "src"
            collated_batch["{}_{}".format(prefix, k)] = v

        return collated_batch

    def collate_fn(self, batch):
        collated_batch = {}

        for attr_name in "text", "target":
            if attr_name == "text":
                # ~~~ Tokenize, truncate and pad the input text ~~~
                max_length = self.params.get("max_input_length", None)
                tokenizer_output = self.tokenizer(
                    [sample[attr_name] for sample in batch],
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=self.params["padding"],
                    max_length=max_length,
                    truncation=self.params["truncation"],
                )
            elif attr_name == "target":
                # ~~~ Truncate and pad the target ids ~~~
                max_length = self.params.get("max_output_length", None)
                input_ids = [sample[f"{attr_name}_ids"][:max_length] for sample in batch]
                tokenizer_output = self.tokenizer.pad(
                    {"input_ids": input_ids}, return_tensors="pt", padding="longest", return_attention_mask=True
                )
            else:
                raise Exception("Unexpected attribute name `{}`!".format(attr_name))

            for k, v in tokenizer_output.items():
                if attr_name == "text":
                    prefix = "src"
                elif attr_name == "target":
                    prefix = "tgt"
                else:
                    Exception("Unexpected attribute name `{}`!".format(attr_name))

                collated_batch["{}_{}".format(prefix, k)] = v

        collated_batch["id"] = [sample["id"] for sample in batch]

        if self.params.get("target_padding_token_id", None) is not None:
            tgt_input_ids = collated_batch["tgt_input_ids"]
            tgt_input_ids.masked_fill_(
                tgt_input_ids == self.tokenizer.pad_token_id, self.params["target_padding_token_id"]
            )

        collated_batch["raw"] = batch

        return collated_batch
