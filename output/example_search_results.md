# Example Search Results

Output of:
```bash
python app.py search "beauty creators who do skincare tutorials with ingredient breakdowns" --top-k 8
```

```
🔍  Top 8 results for: "beauty creators who do skincare tutorials with ingredient breakdowns"

╭─────────┬──────────────────┬────────────┬──────────┬─────────────┬────────┬───────────╮
│ Score   │ Username         │ Platform   │ Niche    │ Followers   │ Eng %  │ Country   │
├─────────┼──────────────────┼────────────┼──────────┼─────────────┼────────┼───────────┤
│ 0.782   │ @glowwithrae     │ tiktok     │ beauty   │ 487.0K      │ 9.2%   │ US        │
│ 0.741   │ @skincarebysana  │ instagram  │ beauty   │ 412.0K      │ 9.8%   │ CA        │
│ 0.719   │ @dermdoctor_amy  │ instagram  │ beauty   │ 534.0K      │ 6.4%   │ US        │
│ 0.687   │ @asmr_skincare_vi│ tiktok     │ beauty   │ 445.0K      │ 9.5%   │ US        │
│ 0.634   │ @beautymarks_di  │ youtube    │ beauty   │ 756.0K      │ 7.8%   │ US        │
│ 0.601   │ @curlqueen_nat   │ youtube    │ beauty   │ 445.0K      │ 8.5%   │ US        │
│ 0.578   │ @glam_by_destiny │ tiktok     │ beauty   │ 623.0K      │ 10.3%  │ US        │
│ 0.542   │ @nailart_rosa    │ tiktok     │ beauty   │ 534.0K      │ 10.8%  │ US        │
╰─────────┴──────────────────┴────────────┴──────────┴─────────────┴────────┴───────────╯
```

---

Output with filters:
```bash
python app.py search "high protein meal ideas for gym" --top-k 5 --niche food --min-engagement 9.0
```

```
🔍  Top 5 results for: "high protein meal ideas for gym"

╭─────────┬────────────────────┬────────────┬────────┬─────────────┬────────┬───────────╮
│ Score   │ Username           │ Platform   │ Niche  │ Followers   │ Eng %  │ Country   │
├─────────┼────────────────────┼────────────┼────────┼─────────────┼────────┼───────────┤
│ 0.714   │ @proteinchef_sam   │ tiktok     │ food   │ 356.0K      │ 9.6%   │ US        │
│ 0.621   │ @organicmama_jo    │ tiktok     │ food   │ 345.0K      │ 11.3%  │ US        │
│ 0.583   │ @bakewithanna      │ tiktok     │ food   │ 412.0K      │ 11.7%  │ US        │
│ 0.541   │ @smoothie_queen    │ tiktok     │ food   │ 198.0K      │ 12.4%  │ US        │
│ 0.498   │ @spicymaya_eats    │ tiktok     │ food   │ 298.0K      │ 10.4%  │ US        │
╰─────────┴────────────────────┴────────────┴────────┴─────────────┴────────┴───────────╯
```

> **Note:** Similarity scores are cosine similarity between the query embedding and each
> creator's bio/niche embedding (all-MiniLM-L6-v2). Higher = more relevant. Scores above
> 0.6 typically indicate strong semantic matches.
