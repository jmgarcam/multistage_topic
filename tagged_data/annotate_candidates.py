#!/usr/bin/env python3
"""Materializa el etiquetado manual de los 2.000 textos candidatos.

Las decisiones se tomaron leyendo exclusivamente el texto de cada candidato y
las definiciones de ``temas.json``. El script no consulta corpus etiquetados,
resultados de modelos ni anotaciones anteriores.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / 'candidate_tagged_data' / 'candidate_tagged_data.json'
TOPICS_FILE = PROJECT_DIR / 'temas.json'
OUTPUT_FILE = PROJECT_DIR / 'tagged_data' / 'tagged_data.json'
NON_NONE_OUTPUT_FILE = PROJECT_DIR / 'tagged_data' / 'tagged_data_no_none.json'
REPORT_FILE = PROJECT_DIR / 'tagged_data' / 'tagging_report.json'
EXPECTED_INPUT_SHA256 = (
    'be74887d43901e41da09199b64969c7ddc8851b452f09e8314b191f2d46c820e'
)
EXPECTED_RECORDS = 2_000
NONE_TOPIC = 'ninguno'

# Abreviaturas usadas únicamente para compactar la tabla manual. Todo índice
# que no aparece en RAW_MANUAL_LABELS fue revisado y se etiqueta como ninguno.
TOPIC_CODES = {
    'd': 'deportes',
    'h': 'salud y medicina',
    't': 'tecnología e informática',
    'e': 'cine, música y entretenimiento',
    'u': 'educación',
    's': 'ciencia y espacio',
    'f': 'gastronomía y alimentación',
    'r': 'turismo y viajes',
    'a': 'automoción',
    'g': 'agricultura y ganadería',
    'v': 'vivienda y construcción',
    'm': 'moda y belleza',
    'n': 'animales y naturaleza',
    'w': 'meteorología y fenómenos naturales',
    'i': 'religión y espiritualidad',
}

RAW_MANUAL_LABELS = """
0=e 1=m 4=t 5=w 8=w 10=e 13=a 15=f 18=s+h 20=s 23=e 27=f 28=v 31=t 32=e 33=u 35=m 36=a 37=a 39=a 41=t 42=e 46=d 50=v 52=g 53=h 55=r 58=t 64=a 67=f 68=d 71=i 73=v 78=t 83=d 84=v 85=e 86=v 88=f 95=d
100=d 102=d 106=i 107=d 110=h 113=a 115=v 122=v 123=v 124=m 128=e 134=v 136=w+h 138=d 139=e 140=e 141=e 143=e 145=v 149=v 155=h 157=e 161=v 165=s 166=a 170=d 172=e 173=u 174=d 175=v 176=v 177=e 178=v 181=t 184=v 187=w 188=d 191=s 192=w 198=e
200=e 201=e 204=t 205=h 207=i+h 208=e 210=s 213=w 215=u 219=d 220=t 225=v 226=t 227=v 232=g 233=v 243=d 244=t 245=u 246=e 247=e 249=d 250=e 252=a 253=v 265=d+h 266=t 267=v 269=r 272=t 273=v 276=v 281=w 294=s 295=e 298=i 299=v
300=e 301=d 302=w 303=a 304=h 305=v 307=u 308=v 313=e 319=f+h 320=i 321=u 322=e 324=n 325=w 328=s+h 329=u 330=v 331=v 332=n 334=n 337=e 338=v 340=w 341=t 343=v 344=h 347=v 348=h 351=v 352=h 357=v 361=n 365=v 366=v 367=u 371=w 373=d 375=d 379=v 383=w 384=v 386=v 387=d 391=t 392=t 393=d 394=v 395=w 397=t
400=m 401=f 405=d 407=g 409=a 410=v 415=m 417=t 419=s 421=d+e 423=h 426=v 427=d 429=f 434=w 436=e 439=v 440=v 442=v 443=m 450=e 452=e 453=f 456=e 457=v 460=a 464=h 465=h 467=e 471=t 475=a 478=w 479=v 483=v 484=v 486=e 487=v 493=s 496=v 497=a
502=a 505=t 509=d 510=v 513=d 514=h 519=v 520=u 522=d 523=a 524=v 527=v 530=h 533=e 535=h 538=e 540=v 544=v 545=w 546=e 551=e 552=r 555=v 558=i 561=e 562=e 567=u 571=f 574=f 580=d 581=d 583=v 589=v 590=h 592=s 594=e 599=t
600=t 604=t 608=f 609=w 611=f+h 614=t 616=a 617=d 620=d 621=u 625=u 626=h 629=t 632=e 633=a 635=f 637=v 639=h 645=d 646=s+t 648=f 651=h 654=w 656=a 661=t 662=u 663=u 665=v 666=v 667=a 668=h 672=v 673=e 677=e 680=w 684=h 687=d 689=n 691=a 693=e 697=d
703=v 705=e 706=h 708=t 710=v 711=u 720=v 723=v 724=i 725=d 728=t 729=a 732=a 734=d 737=i 739=i 741=d 742=a 743=v 747=v 748=v 749=v 754=v 758=s 759=t 762=v 765=e 766=e 771=r 772=v 773=u 779=v 781=f 782=w 784=d 789=h 790=v 794=e+t 799=e
805=f 807=d 808=v 810=v 812=h 814=d 816=s 818=t 821=t 822=h 823=i+h 824=d 825=w 826=u 829=v 834=v 835=e 836=e 837=e 838=t+e 839=d 840=w+h 841=g 843=e 844=t 845=u 849=d 850=v 856=e 857=v 858=n 861=e 862=r 864=d 865=f 866=d+w 869=r 870=g 872=d 873=v 874=e 875=v 885=s 893=n 894=a 898=d
900=v 901=e 906=w 907=e 909=v 910=v 911=a 912=w 913=u 917=e 918=e 922=h 926=t 927=a 929=i+h 930=h 931=v 932=a 933=i 949=u 952=w 954=w 957=v 961=t 962=u 963=e 964=w 967=u 968=d 969=v 970=a 972=u 975=w 980=a 981=a 985=d 986=w 993=v 994=v 995=v 996=v 998=h+m
1000=e 1003=h 1005=h 1007=g 1008=e 1015=e 1017=w 1019=v 1020=e 1021=e 1023=v 1025=i 1026=r 1030=i 1031=m 1036=h 1038=e 1040=e 1045=v 1047=g 1048=w 1050=w 1053=e 1054=i 1055=f 1058=v 1059=e 1060=d 1061=e 1063=g 1065=w 1066=e 1067=e 1068=v 1070=w 1071=w 1073=d 1076=w 1078=v 1080=d 1082=u 1083=r 1086=d 1088=v 1089=e 1090=t 1091=t 1092=e 1094=h 1097=d 1099=v
1105=t+h 1106=e 1113=h 1114=d 1117=v 1119=v 1121=g 1122=e 1124=w 1126=g 1127=a 1128=i 1130=v 1134=d 1136=v 1137=e 1139=e 1143=v 1145=h 1150=v 1154=f 1157=e 1159=m 1165=e 1167=v 1169=v 1171=v 1172=f 1173=v 1176=e 1177=e 1183=v 1184=h 1185=h 1187=v 1188=m 1189=v 1190=w 1191=i 1192=m 1195=w 1198=v
1201=e 1202=d 1203=d 1208=h 1209=h 1214=v 1222=u 1225=t 1235=t 1239=t 1245=t 1249=w 1250=i 1251=i 1252=f 1253=h 1256=v 1257=e 1258=d 1260=e 1261=h 1263=h 1266=t 1267=d 1269=t 1270=d 1271=w 1273=h 1279=v 1285=v 1289=e 1291=a 1293=t 1297=v 1298=u
1301=t 1306=v 1308=f 1309=v 1312=e+t 1315=h 1316=v 1319=d 1321=u 1323=e 1326=t 1328=h+m 1330=a 1332=w+a 1336=e 1338=d 1340=r 1343=h+f 1350=e 1352=n 1353=h 1356=e 1357=i 1358=t 1362=e 1364=t+h 1368=v 1369=e 1371=t 1373=v 1374=d 1378=r 1379=h 1382=u 1394=v 1395=t 1397=h 1398=a 1399=h
1402=e 1403=h 1405=t 1416=e 1418=f+h 1423=v 1425=e 1426=v 1427=d 1433=f+h 1435=a 1436=r 1438=v 1439=a 1448=e 1449=h 1450=e 1452=h 1454=f+h 1458=s+h 1459=d 1462=d 1463=d 1465=e 1468=v 1470=v 1474=d 1480=h 1484=h 1492=v 1494=m
1502=v 1504=e 1505=i+t 1511=e 1512=d 1513=e 1516=v 1517=f 1518=v 1519=v 1520=v 1521=v 1522=m 1523=v 1524=i 1526=v 1527=d 1528=s 1530=h 1537=v 1539=h 1541=w 1542=d 1543=v 1545=w 1547=t 1554=h 1559=w 1560=u 1561=e 1562=v 1564=e 1565=a 1566=m 1567=e 1569=h 1573=v 1575=v 1577=e 1583=e 1587=e 1591=i+h 1592=e 1593=e 1597=t 1598=e 1599=d
1600=h 1605=e 1607=s 1614=d 1615=e 1617=v 1618=v 1620=u 1622=w 1623=t 1624=d 1627=e 1628=v 1630=d 1632=f+h 1633=e 1634=v 1636=v 1641=w 1646=s 1647=s 1648=e 1658=d 1661=a 1663=t 1664=d 1667=a 1668=d 1669=u 1672=d 1674=t 1675=v 1677=i+v 1678=t 1679=e 1681=e 1682=e 1683=v 1685=w 1686=f+n 1687=d 1689=v 1693=e 1694=d 1698=i
1700=e 1701=u 1702=i 1705=v 1709=m 1710=a 1714=i 1718=f 1720=v 1724=f 1725=d 1726=v 1727=a 1729=t 1732=e 1734=d 1738=e 1745=e 1747=v 1756=t 1760=d 1761=w 1764=d 1766=v 1767=e 1769=e 1770=d 1771=s 1772=v 1773=u 1774=e 1777=e 1779=a 1781=v 1783=v 1785=w 1786=v 1788=h 1789=u 1792=u 1795=v 1798=a 1799=d+h
1801=v 1803=a 1805=v 1806=t 1809=e 1810=e 1811=s 1812=w 1813=t 1816=d 1818=t 1824=r 1827=t 1828=t 1830=a 1832=v 1834=t 1838=e 1849=f 1850=v 1855=h 1860=i 1861=u 1863=h 1864=w 1867=i+v 1870=v 1875=a 1877=u 1880=a 1882=v 1886=a+t 1888=t 1890=v 1891=a 1892=t+h
1900=a 1902=d+e 1911=d 1912=r 1913=v 1917=d 1918=v 1919=w 1920=d 1922=d 1924=e 1927=g 1929=d 1930=v 1931=e 1933=w 1938=u 1940=e 1943=a 1944=v 1947=e 1948=e 1952=w 1954=v 1955=e 1957=v 1958=v 1960=t 1961=e 1963=t 1965=v 1968=v 1972=v 1973=u 1974=e 1980=w 1985=v 1987=u 1990=h 1992=w 1994=d
"""

LOW_CONFIDENCE_INDICES = {
    25, 38, 91, 197, 209, 222, 282, 292, 362, 416, 707, 775, 871,
    944, 1004, 1027, 1310, 1349, 1471, 1473, 1549, 1582, 1717, 1757,
    1879, 1897, 1966,
}

JUSTIFICATION_FOCUS = {
    'deportes': 'una actividad o competición deportiva',
    'salud y medicina': 'una cuestión sanitaria o médica',
    'tecnología e informática': 'una tecnología o servicio digital',
    'cine, música y entretenimiento': 'una actividad cultural o de entretenimiento',
    'educación': 'la enseñanza, el alumnado o el sistema educativo',
    'ciencia y espacio': 'una investigación o fenómeno científico',
    'gastronomía y alimentación': 'la alimentación o la gastronomía',
    'turismo y viajes': 'los viajes, alojamientos o la actividad turística',
    'automoción': 'los vehículos, la conducción o la industria automovilística',
    'agricultura y ganadería': 'la producción o actividad agraria',
    'vivienda y construcción': 'la vivienda, las obras o el mercado residencial',
    'moda y belleza': 'la moda, la belleza o el cuidado estético',
    'animales y naturaleza': 'los animales, la fauna o su conservación',
    'meteorología y fenómenos naturales': 'un fenómeno meteorológico o geológico',
    'religión y espiritualidad': 'una institución, práctica o acontecimiento religioso',
}


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as file_handle:
        return json.load(file_handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_manual_labels() -> dict[int, list[str]]:
    annotations: dict[int, list[str]] = {}
    for token in RAW_MANUAL_LABELS.split():
        raw_index, raw_codes = token.split('=', maxsplit=1)
        index = int(raw_index)
        if index in annotations:
            raise ValueError(f'Índice manual duplicado: {index}')
        annotations[index] = [TOPIC_CODES[code] for code in raw_codes.split('+')]
    return annotations


def justification(labels: list[str]) -> str:
    if labels == [NONE_TOPIC]:
        return (
            'El acontecimiento principal no encaja de forma suficiente en '
            'ninguno de los temas definidos.'
        )
    if len(labels) == 1:
        return (
            f'El acontecimiento principal se centra en '
            f'{JUSTIFICATION_FOCUS[labels[0]]}, conforme a la definición de '
            f'«{labels[0]}».'
        )
    quoted = ' y '.join(f'«{label}»' for label in labels)
    return f'El texto aborda de forma central y explícita {quoted}.'


def main() -> None:
    input_hash = file_sha256(INPUT_FILE)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            'candidate_tagged_data.json ha cambiado desde la revisión manual; '
            'es necesario revisar de nuevo las decisiones antes de continuar.'
        )

    candidates = load_json(INPUT_FILE)
    topics = load_json(TOPICS_FILE)
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_RECORDS:
        raise SystemExit(f'Se esperaban exactamente {EXPECTED_RECORDS} candidatos.')
    if not isinstance(topics, dict) or not topics:
        raise SystemExit('temas.json debe contener un objeto no vacío.')

    annotations = parse_manual_labels()
    invalid_indices = sorted(set(annotations) - set(range(len(candidates))))
    if invalid_indices:
        raise SystemExit(f'Índices manuales fuera de rango: {invalid_indices}')

    allowed_topics = set(topics) | {NONE_TOPIC}
    tagged: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise SystemExit(f'Candidato {index}: se esperaba un objeto.')

        labels = annotations.get(index, [NONE_TOPIC])
        if not labels or any(label not in allowed_topics for label in labels):
            raise SystemExit(f'Candidato {index}: etiquetas no válidas {labels}.')
        if NONE_TOPIC in labels and len(labels) != 1:
            raise SystemExit(
                f'Candidato {index}: ninguno no puede combinarse con otro tema.'
            )

        annotated = dict(candidate)
        annotated['tema'] = labels
        annotated['confianza'] = (
            'baja' if index in LOW_CONFIDENCE_INDICES else 'media'
        )
        annotated['justificacion'] = justification(labels)
        tagged.append(annotated)

    non_none_tagged = [
        record for record in tagged if record['tema'] != [NONE_TOPIC]
    ]
    save_json(OUTPUT_FILE, tagged)
    save_json(NON_NONE_OUTPUT_FILE, non_none_tagged)

    topic_counts = Counter(label for record in tagged for label in record['tema'])
    confidence_counts = Counter(record['confianza'] for record in tagged)
    report = {
        'status': 'complete',
        'input_file': str(INPUT_FILE.resolve()),
        'input_sha256': input_hash,
        'topics_file': str(TOPICS_FILE.resolve()),
        'output_file': str(OUTPUT_FILE.resolve()),
        'non_none_output_file': str(NON_NONE_OUTPUT_FILE.resolve()),
        'annotation_method': (
            'Revisión manual del texto de cada candidato frente a las '
            'definiciones de temas.json.'
        ),
        'annotation_sources': ['candidate_text', 'temas.json'],
        'labelled_examples_used': False,
        'total_records': len(tagged),
        'manually_reviewed_records': len(tagged),
        'none_records': topic_counts[NONE_TOPIC],
        'non_none_records': len(non_none_tagged),
        'multi_topic_records': sum(len(record['tema']) > 1 for record in tagged),
        'topic_counts': dict(sorted(topic_counts.items())),
        'confidence_counts': dict(sorted(confidence_counts.items())),
    }
    save_json(REPORT_FILE, report)

    print(f'Registros etiquetados: {len(tagged)}')
    print(f'Registros con ninguno: {topic_counts[NONE_TOPIC]}')
    print(f'Registros distintos de ninguno: {len(non_none_tagged)}')
    print(f'Registros multietiqueta: {report["multi_topic_records"]}')
    print(f'Resultado: {OUTPUT_FILE.resolve()}')
    print(f'Resultado sin ninguno: {NON_NONE_OUTPUT_FILE.resolve()}')
    print(f'Informe: {REPORT_FILE.resolve()}')


if __name__ == '__main__':
    main()
