"jieba_link_extractor unit test"
import pytest
from langchain_community.graph_vectorstores.extractors.jieba_link_extractor import (
    JiebaLinkExtractor
)
from langchain_community.graph_vectorstores.links import Link

CONTENT1 = '''
文洁默默地离开了已经空无一人一片狼藉的操场，走上回家的路。当她走到教工宿舍楼下时，听到了从二楼自家窗口传出的一阵阵痴笑声，
这声音是那个她曾叫做妈妈的女人发出的。文洁默默地转身走去，任双脚将她带向别处。
她最后发现自己来到了阮雯的家门前，在大学四年中，阮老师一直是她的班主任，也是她最亲密的朋友。
在叶文洁读天体物理专业研究生的两年里，再到后来停课闹革命至今，阮老师一直是她除父亲外最亲近的人。
阮雯曾留学剑桥，她的家曾对叶文洁充满了吸引力，那里有许多从欧洲带回来的精致的书籍、油画和唱片，一架钢琴；还有一排放在精致小木架上的欧式烟斗，
父亲那只就是她送的，这些烟斗有地中海石楠根的，有土耳其海泡石的，每一个都仿佛浸透了曾将它们拿在手中和含在嘴里深思的那个男人的智慧，
但阮雯从未提起过他。这个雅致温暖的小世界成为文洁逃避尘世风暴的港湾。但那是阮雯的家被抄之前的事，她在运动中受到的冲击和文洁父亲一样重，
在批斗会上，红卫兵把高跟鞋挂到她脖子上，用口红在她的脸上划出许多道子，以展示她那腐朽的资产阶级生活方式。
叶文洁推开阮雯的家门，发现抄家后混乱的房间变得整洁了，那几幅被撕的油画又贴糊好挂在墙上，歪倒的钢琴也端正地立在原位，虽然已被砸坏不能弹了，
但还是擦得很干净，残存的几本精装书籍也被整齐地放回书架上……阮雯端坐在写字台前的那把转椅上，安详地闭着双眼。
叶文洁站在她身边，摸摸她的额头、脸和手，都是冰凉的，其实文洁在进门后就注意到了写字台上倒放着的那个已空的安眠药瓶。
她默默地站了一会儿，转身走去，悲伤已感觉不到了，她现在就像一台盖革计数仪，当置身于超量的辐射中时，
反而不再有任何反应，没有声响，读数为零。但当她就要出门时，还是回过头来最后看了阮雯一眼，
她发现阮老师很好地上了妆，她抹了口红，也穿上了高跟鞋。
'''

@pytest.mark.requires("jieba")
def test_tfidf_analyzer():
    "tfidf analyzer test case"
    extractor = JiebaLinkExtractor(analyzer="tfidf")
    results = extractor.extract_one(CONTENT1)
    assert results == {
        Link.bidir(kind="kw", tag="阮雯"),
        Link.bidir(kind="kw", tag="文洁"),
        Link.bidir(kind="kw", tag="叶文洁")
    }

@pytest.mark.requires("jieba")
def test_textrank_analyzer():
    "textrank analyzer test case"
    extractor = JiebaLinkExtractor(analyzer="textrank")
    results = extractor.extract_one(CONTENT1)
    assert results == {
        Link.bidir(kind="kw", tag="阮雯"),
        Link.bidir(kind="kw", tag="文洁"),
        Link.bidir(kind="kw", tag="叶文洁")
    }

@pytest.mark.requires("jieba")
def test_mixed_analyzer():
    "mixed analyzer test case"
    extractor = JiebaLinkExtractor(analyzer="mixed",weight=(0.4,0.6))
    results = extractor.extract_one(CONTENT1)
    assert results == {
        Link.bidir(kind="kw", tag="文洁"),
        Link.bidir(kind="kw", tag="叶文洁"),
        Link.bidir(kind="kw", tag="阮雯")
    }

@pytest.mark.requires("jieba")
def test_with_extract_keywords_kwargs():
    "test with extract_keywords_kwargs params"
    extractor = JiebaLinkExtractor(extract_keywords_kwargs={
        "topK": 5, 
        "withWeight": True, 
        "allowPOS": ('n','nr','ns','nt','nz')
        }
    )
    results = extractor.extract_one(CONTENT1)
    assert results == {
        Link.bidir(kind="kw", tag="文洁"),
        Link.bidir(kind="kw", tag="阮雯"),
        Link.bidir(kind="kw", tag="高跟鞋"),
        Link.bidir(kind="kw", tag="口红"),
        Link.bidir(kind="kw", tag="叶文洁")
    }
