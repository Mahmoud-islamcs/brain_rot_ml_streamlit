"""
Display-layer translation dictionary for Brain Rot Analytics.

This module ONLY contains text shown to the user in the UI (labels, titles,
buttons, messages). It has no effect on the machine learning pipeline,
feature names, or the values sent to/received from the model.
"""

TRANSLATIONS = {
    "en": {
        # Sidebar
        "lang_toggle_button": "العربية 🇪🇬",
        "sidebar_title": "BrainRot Analytics",
        "sidebar_subtitle": "AI-Powered Behavioral Intelligence Platform",
        "nav_predict": "Predict",
        "nav_insights": "Insights",
        "nav_about": "About",
        "sidebar_model_label": "Model",
        "sidebar_records_label": "Records",
        "made_by": "Made by [Mahmoud Islam](https://www.linkedin.com/in/mahmoud-islam-analytics/)",

        # Predict page
        "predict_title": "Mental Health & Wellbeing Predictor",
        "predict_intro": (
            "Enter your daily habits below to predict your current digital "
            "distraction stage (Brain Rot Stage), based on a machine learning "
            "model trained on 5,000 Egyptian student records."
        ),
        "label_reels": "Daily Short-Form Videos Watched (Reels/Shorts/TikTok)",
        "label_coffee": "Daily Caffeinated Drinks Consumed",
        "label_device": "Primary Device Used",
        "label_late_night_question": "Do you use your phone after midnight?",
        "label_yes": "Yes",
        "label_no": "No",
        "label_study_hours": "Daily Effective Study/Productivity Hours",
        "label_focus_sessions": "Daily Deep Focus Sessions (Uninterrupted)",
        "label_age": "Age",
        "device_smartphone": "Smartphone",
        "device_tablet": "Tablet",
        "device_pc": "PC",
        "predict_button": "Execute AI Prediction",
        "spinner_text": "Analyzing your data...",
        "result_title": "Current Digital Distraction Stage",
        "recommendation_label": "Recommendation",
        "proba_title": "Prediction Probabilities",

        # Insights page
        "insights_title": "Model Insights",
        "insights_subtitle": "Performance details and interpretation of the winning model:",
        "feature_importance_title": "Feature Importance",
        "feature_importance_missing": "Run train_model.ipynb first to generate this chart.",
        "confusion_matrix_title": "Confusion Matrix",
        "confusion_matrix_missing": "Run train_model.ipynb first to generate this image.",
        "classification_report_title": "Classification Report",
        "classification_report_missing": "No saved report found. Run train_model.ipynb first.",
        "correlation_title": "Correlation Analysis",
        "correlation_heatmap_title": "Correlation Heatmap",

        # About page
        "about_title": "About Brain Rot Analytics",
        "about_project_heading": "About the Project",
        "about_project_text": (
            '<b>"Brain Rot Analytics"</b> is a comprehensive Data Science and Machine '
            "Learning graduation project designed to analyze digital wellbeing. "
            "It studies the profound relationship between daily behavioral habits"
            "\u2014specifically short-form video consumption\u2014and student productivity. "
            "The project delivers a predictive model that successfully classifies a "
            "student's digital distraction level into one of four distinct stages: "
            "<span style='color:#10B981; font-weight:bold;'>Healthy</span>, "
            "<span style='color:#3B82F6; font-weight:bold;'>Casual</span>, "
            "<span style='color:#F59E0B; font-weight:bold;'>Advanced</span>, or "
            "<span style='color:#EF4444; font-weight:bold;'>Critical</span>."
        ),
        "about_dataset_heading": "About the Dataset & Preprocessing",
        "about_dataset_text": (
            "The model is built on a comprehensive dataset containing "
            "<b>5,000 rows and 30 columns</b>. To preserve full sample size and "
            "statistical integrity, missing values found across 51 rows (in "
            "features like Age, Region, and Device Type) were meticulously "
            "imputed using median and mode strategies rather than being discarded."
        ),
        "about_benefits_heading": "Key Project Benefits",
        "about_benefits": [
            "Detects early indicators of digital distraction before escalating to critical stages.",
            "Provides personalized, actionable recommendations to mitigate screen-time harms.",
            "Empowers awareness campaigns with robust, data-driven insights into modern digital habits.",
            "Pinpoints specific destructive behaviors (e.g., late-night usage, short-video loops) that heavily impair cognitive focus.",
        ],
        "about_committee_heading": "How to Explain the Model to the Committee",
        "about_committee_p1_heading": "1) Machine Learning Pipeline & Models Evaluated:",
        "about_committee_p1": (
            "We implemented a comparative benchmark across three diverse algorithms: "
            "<b>Logistic Regression</b> (as our baseline model), <b>Random Forest</b>, "
            "and <b>XGBoost</b>. Every model was optimized utilizing "
            "<code>RandomizedSearchCV</code> for hyperparameter tuning, and strict "
            "class balancing was enforced via <code>class_weight='balanced'</code> "
            "alongside a stratified train/test split to address the highly "
            "imbalanced class distribution (Healthy: 61%, Critical: 18%, "
            "Advanced: 12.5%, Casual: 8.5%)."
        ),
        "about_committee_p2_heading": "2) Rigorous Metric Selection & Winning Model:",
        "about_committee_p2": (
            "The <b>Random Forest Classifier</b> emerged as the champion model, "
            "achieving an outstanding <b>Macro F1-score of 0.903</b> and an "
            "<b>Overall Accuracy of 94.5%</b>. We explicitly relied on the Macro "
            "F1-score as our primary evaluation metric rather than accuracy alone, "
            "ensuring that the model performs equally well on the minority classes "
            "(Casual and Advanced) and is robust against data imbalance."
        ),
        "about_committee_p3_heading": "3) Top Feature Importances:",
        "about_committee_p3": (
            "The model's decisions are highly transparent and logical. Inline "
            "Feature Importance analysis indicates that the top 3 predictive "
            "drivers are: <code>Total_Reels_Watched</code> &rarr; "
            "<code>Focus_Sessions_Count</code> &rarr; <code>Study_Hours</code>. "
            "This perfectly validates our core hypothesis regarding digital "
            "distraction and academic focus."
        ),
        "about_leakage_heading": "Prevention of Data Leakage",
        "about_leakage_text": (
            "To guarantee genuine generalization, strict feature selection was "
            "conducted. Only <b>7 behavioral features</b> were allowed into "
            "training (<i>Age, Total_Reels_Watched, Coffee_Consumed_Per_Day, "
            "Focus_Sessions_Count, Study_Hours, Is_Late_Night, Device_Type</i>). "
            "Target-derived or highly correlated columns\u2014such as "
            "<code>Brainrot_Exposure_Score</code>, <code>Wellbeing_Score</code>, "
            "<code>Attention_Span_Level</code>, <code>Aura_Color_Code</code>, "
            "<code>Coffee_Level</code>, and <code>Smoking_Status</code>\u2014were "
            "<b>deliberately excluded</b> from the training phase. This prevents "
            "mathematical data leakage, ensuring that the dashboard's performance "
            "is realistic and not artificially inflated."
        ),
    },

    "arz": {
        # Sidebar
        "lang_toggle_button": "English 🇬🇧",
        "sidebar_title": "برين روت أناليتكس",
        "sidebar_subtitle": "منصة ذكاء اصطناعي لتحليل السلوك الرقمي والإنتاجية",
        "nav_predict": "قيس تشتتك",
        "nav_insights": "تحليلات الموديل",
        "nav_about": "عن المشروع",
        "sidebar_model_label": "الموديل المستخدم",
        "sidebar_records_label": "حجم عينة الدراسة",
        "made_by": "صُنع بواسطة [محمود إسلام](https://www.linkedin.com/in/mahmoud-islam-analytics/)",

        # Predict page
        "predict_title": "مستكشف التشتت والروقان الرقمي",
        "predict_intro": (
            "شاركونا عاداتكم اليومية البسيطة دي عشان نعرف مرحلة التشتت الرقمي بتاعتكم دلوقتي "
            "(Brain Rot Stage)، بالاعتماد على موديل ذكاء اصطناعي متدرب على بيانات حقيقية لـ 5000 طالب مصري."
        ),
        "label_reels": "بتتفرج على كام فيديو قصير في اليوم؟ (ريلز/شورتس/تيك توك)",
        "label_coffee": "بتشرب كام فنجان قهوة أو مشروب كافيين في اليوم؟",
        "label_device": "إيه جهازك الأساسي اللي بتسرح قدامه؟",
        "label_late_night_question": "بتسهر على الموبايل تحت السرير بعد نص الليل؟",
        "label_yes": "أيوة وبندم تاني يوم",
        "label_no": "لأ بنام بدري",
        "label_study_hours": "بتنجز كام ساعة مذاكرة أو شغل فعلي في اليوم؟",
        "label_focus_sessions": "بتعمل كام جلسة تركيز عميق (من غير أي تشتت أو إشعارات)؟",
        "label_age": "عمرك كام سنة يا فنان؟",
        "device_smartphone": "موبايل",
        "device_tablet": "تابلت",
        "device_pc": "كمبيوتر / لابتوب",
        "predict_button": "احسبلي النتيجة بسرعة!",
        "spinner_text": "بندرس عاداتك السلوكية.. ثواني يا فنان...",
        "result_title": "مرحلة التشتت الرقمي الحالية",
        "recommendation_label": "نصيحة أخوية سريعة",
        "proba_title": "نسبة ثقة الموديل في كل مرحلة",

        # Insights page
        "insights_title": "كواليس الموديل (Model Insights)",
        "insights_subtitle": "تفاصيل الأداء وإزاي الموديل الفائز بياخد قراراته:",
        "feature_importance_title": "أهم العوامل المؤثرة في القرار",
        "feature_importance_missing": "شغّل ملف train_model.ipynb الأول عشان ترسم التشارت ده هنا.",
        "confusion_matrix_title": "مصفوفة الالتباس (Confusion Matrix)",
        "confusion_matrix_missing": "شغّل ملف train_model.ipynb الأول عشان نظهر الصورة دي هنا.",
        "classification_report_title": "تقرير دقة التصنيف (Classification Report)",
        "classification_report_missing": "مفيش تقرير محفوظ حالياً. شغّل ملف train_model.ipynb الأول.",
        "correlation_title": "تحليل الارتباط بين العوامل المختلفة",
        "correlation_heatmap_title": "الخريطة الحرارية للارتباط (Correlation Heatmap)",

        # About page
        "about_title": "حكاية مشروع Brain Rot Analytics",
        "about_project_heading": "قصة وفكرة المشروع",
        "about_project_text": (
            'مشروع <b>"Brain Rot Analytics"</b> هو مشروع تخرج متكامل في مجالي علم البيانات والتعلم الآلي، '
            "بيهدف لتقديم فهم علمي دقيق لأثر العالم الرقمي على حياتنا. المشروع بيدرس بالتفصيل العلاقة العميقة "
            "بين عاداتنا اليومية\u2014وخاصة إدمان الفيديوهات القصيرة السريعة\u2014ومدى تأثيرها على تركيز وإنتاجية الطلاب. "
            "قدرنا نبني موديل ذكاء اصطناعي بيصنف مستوى التشتت لأربع مراحل بدقة عالية: "
            "<span style='color:#10B981; font-weight:bold;'>سليم ومصحصح</span>، "
            "<span style='color:#3B82F6; font-weight:bold;'>سرحان خفيف</span>، "
            "<span style='color:#F59E0B; font-weight:bold;'>تشتت متقدم</span>، أو "
            "<span style='color:#EF4444; font-weight:bold;'>مرحلة الطوارئ الحرجة</span>."
        ),
        "about_dataset_heading": "عن البيانات والمعالجة الإحصائية",
        "about_dataset_text": (
            "الموديل اتدرب على داتا ضخمة وقوية فيها <b>5000 صف و30 عمود</b>. وعشان نحافظ على مصداقية الأرقام "
            "ومتانة التحليل الإحصائي، اتعاملنا بحرص شديد مع القيم المفقودة في 51 صف (زي السن، والمنطقة، ونوع الجهاز) "
            "وعملنالها تعويض ذكي باستخدام الـ median والـ mode بدل ما نستسهل ونمسحها."
        ),
        "about_benefits_heading": "إيه الفايدة اللي طالعة من المشروع؟",
        "about_benefits": [
            "بيكتشف بوادر التشتت الرقمي بدري جداً قبل ما تخرج عن السيطرة ونخش في الزون الحرجة.",
            "بيقدّم نصائح عملية ومخصصة لكل مستخدم بناءً على عاداته الفعلية عشان يحسن يومه.",
            "بيدعم صناع القرار وحملات التوعية بأرقام دقيقة وإحصائيات واقعية من مجتمع الطلاب.",
            "بيشاور بوضوح على السلوكيات الأكثر تدميراً للتركيز (زي سهرة الموبايل واللف غير المنتهي في السوشيال ميديا).",
        ],
        "about_committee_heading": "دليل مناقشة المشروع قدام اللجنة",
        "about_committee_p1_heading": "1) الـ Pipeline والموديلات اللي قارنا بينها:",
        "about_committee_p1": (
            "عملنا تجارب ومقارنة شاملة بين 3 خوارزميات مختلفة: <b>Logistic Regression</b> "
            "كخط أساس للمقارنة، و <b>Random Forest</b>، و <b>XGBoost</b>. كل موديل "
            "أخد حقه بالكامل في الـ Hyperparameter Tuning باستخدام <code>RandomizedSearchCV</code>، "
            "وكمان فرضنا توازن كامل لنسب الفئات في التدريب باستخدام <code>class_weight='balanced'</code> "
            "مع تقسيم Stratified للداتا عشان نعالج عدم توازن العينات "
            "(Healthy: 61%، Critical: 18%، Advanced: 12.5%، Casual: 8.5%)."
        ),
        "about_committee_p2_heading": "2) اختيار المقياس الرياضي والموديل الكسبان:",
        "about_committee_p2": (
            "موديل الـ <b>Random Forest</b> كان هو البطل الفائز بأعلى أداء؛ وحقق "
            "<b>Macro F1-score وصل لـ 0.903</b> و <b>دقة كلية (Accuracy) بلغت 94.5%</b>. "
            "والمهم هنا إننا وضحنا للجنة اعتمادنا على الـ Macro F1-score كمقياس أساسي عشان نضمن "
            "إن الموديل بيصنف الفئات الأقل عدداً (زي الـ Casual والـ Advanced) بنفس الكفاءة ومن غير تحيز للفئة الكبيرة."
        ),
        "about_committee_p3_heading": "3) أهم محركات قرار الموديل:",
        "about_committee_p3": (
            "قرارات الموديل منطقية وشفافة جداً وقابلة للتفسير العلمي؛ تحليل الـ Feature Importance بيأكد "
            "إن أهم 3 عوامل مؤثرة بالترتيب هي: <code>Total_Reels_Watched</code> ثم "
            "<code>Focus_Sessions_Count</code> وأخيراً <code>Study_Hours</code>. "
            "وده بيثبت بالدليل القاطع فرضيتنا الأساسية للمشروع."
        ),
        "about_leakage_heading": "حماية الموديل من تسريب البيانات (Data Leakage)",
        "about_leakage_text": (
            "عشان نضمن إن الموديل حقيقي وبيشتغل في الواقع مش بس في بيئة التدريب المثالية، عملنا فلترة صارمة جداً "
            "للفيتشرز واستبعدنا تماماً أي كولوم مشتق أو مرتبط بالـ target بشكل مباشر "
            "(زي <code>Brainrot_Exposure_Score</code>، و <code>Wellbeing_Score</code>، و <code>Attention_Span_Level</code>) "
            "واكتفينا فقط بـ <b>7 فيتشرز سلوكية نقية</b>. دي حركة هندسية ممتازة تمنع الـ Data Leakage وتثبت "
            "أمام لجنة المناقشة احترافية بيئة العمل الهندسية للمشروع."
        ),
    },
}

STAGE_LABELS = {
    "en": {
        "Healthy": "Healthy",
        "Casual": "Casual",
        "Advanced": "Advanced",
        "Critical": "Critical",
    },
    "arz": {
        "Healthy": "سليم ومصحصح 😎",
        "Casual": "سرحان خفيف ☕",
        "Advanced": "تشتت متقدم 🌀",
        "Critical": "مرحلة الطوارئ الحرجة 🚨",
    },
}

STAGE_RECOMMENDATIONS = {
    "en": {
        "Healthy": "Keep up this healthy pattern. You have an excellent balance between screen time, focus sessions, and study hours. Maintain your current routine.",
        "Casual": "Distraction is still at a mild stage. Try gradually reducing your daily short-form video count and add one extra deep-focus session per week.",
        "Advanced": "Digital distraction is starting to noticeably affect your productivity. Consider setting a daily screen-time limit for short-form videos, increasing your deep-focus sessions, and reducing caffeine intake after 6 PM.",
        "Critical": "This is a critical stage that requires immediate attention: reduce short-form video consumption, especially before bed, try the Pomodoro technique to increase focus sessions, and avoid using your phone after midnight. If you feel this is affecting your daily life, it may help to speak with a specialist.",
    },
    "arz": {
        "Healthy": "عاش يا بطل، استمر على الروقان ده! عندك توازن مثالي بين السوشيال ميديا وجلسات التركيز والمذاكرة. حافظ على يومك الجميل ده.",
        "Casual": "التشتت لسه تحت السيطرة وفي الأول. حاول تقلل فرجتك على الريلز شوية بالتدريج وادخل في جلسة تركيز زيادة كل أسبوع عشان تضبط الدنيا.",
        "Advanced": "التشتت الرقمي بدأ يسحبك بجد ويأثر على يومك وإنتاجيتك. جرب تحدد لنفسك وقت معين للريلز في اليوم، وزوّد جلسات الفوكاس، وحاول تفصل الكافيين بعد 6 المغرب.",
        "Critical": "دي مرحلة طوارئ محتاجة وقفة فورية مع نفسك: ابعد التليفون برة الأوضة قبل ما تنام، جرب تكنيك البومودورو في الشغل، واقفل الموبايل تماماً بعد نص الليل. لو حاسس إن الموضوع مأثر على حياتك بشكل كبير، متترددش إنك تاخد رأي متخصص ينظم معاك يومك.",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Returns the translated string for the given key and language.
    Falls back to English, then to the key itself, if not found.
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )