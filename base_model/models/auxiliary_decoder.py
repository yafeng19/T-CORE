import torch.nn as nn

class AuxiliaryAttentionBlock(nn.Module):
    def __init__(self, attn_mul=4, embed_dim=384, num_heads=16):
        super().__init__()
        self.attn_mul = attn_mul
        self.auxiliary_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.auxiliary_layer_norm1 = nn.LayerNorm(embed_dim)
        self.auxiliary_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.auxiliary_layer_norm2 = nn.LayerNorm(embed_dim)
        self.auxiliary_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.auxiliary_layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, aux_crop, stu_crop):
        # shape of aux_crop and stu_crop: (batch_size, seq_len, embed_dim)

        # MultiheadAttention takes in the query, key, value. Here we use stu_crop to attend to aux_crop.
        cross_attn_output, cross_attn_output_weights = self.auxiliary_cross_attn(stu_crop, aux_crop, aux_crop)
        cross_attn_output = self.auxiliary_layer_norm1(self.attn_mul * cross_attn_output + stu_crop) # layer norm with skip connection
        
        # Then we use cross_attn_output to attend to cross_attn_output itself
        self_attn_output, self_attn_output_weights = self.auxiliary_self_attn(cross_attn_output, cross_attn_output, cross_attn_output)
        self_attn_output = self.auxiliary_layer_norm2(self_attn_output + cross_attn_output) # layer norm with skip connection

        # Finally, apply feed forward.
        output = self.auxiliary_linear(self_attn_output)
        output = self.auxiliary_layer_norm3(output + self_attn_output) # layer norm with skip connection
        
        return output, cross_attn_output_weights, self_attn_output_weights



class AuxiliaryDecoder(nn.Module):
    def __init__(self, attn_mul=4, num_blocks=1, embed_dim=384, num_heads=16):
        super().__init__()
        self.decoder = nn.ModuleList([AuxiliaryAttentionBlock(attn_mul, embed_dim, num_heads) for _ in range(num_blocks)])

    def forward(self, aux_crop, stu_crop):
        output = stu_crop
        for block in self.decoder:
            output = block(aux_crop, output)

        return output

