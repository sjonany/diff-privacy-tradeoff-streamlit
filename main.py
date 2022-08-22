import streamlit as st
import pandas as pd
import snsql
from snsql import Privacy, Stat, Mechanism
import numpy as np
import altair as alt

# Constants
QUERY = """
SELECT binned_age, COUNT(*)
FROM PUMS.PUMS
GROUP BY binned_age
"""
CSV_PATH = 'data/PUMS.csv'
META_PATH = 'data/PUMS.yaml'
ALPHA = 0.05
MIN_EPS_VAL = 0.1
EPS_STEP_SIZE = 0.1
MAX_EPS_VAL = 10.0

def load_data(csv_path):
  data = pd.read_csv(csv_path)
  data['binned_age'] = pd.cut(x=data['age'], bins=[0,20, 40, 60, 80, 100]).astype('str')
  return data

def get_unnoised_result(data):
  return data["binned_age"].value_counts().to_dict()

def get_noised_result(epsilon, data, query, meta_path):
  """
  @param epsilon (double)
  @return Dict<String, double>. Groupby key -> noised total count.
  """
  privacy = Privacy(epsilon=epsilon)
  privacy.mechanisms.map[Stat.count] = Mechanism.laplace
  reader = snsql.from_connection(data, privacy=privacy, metadata=meta_path)
  result = reader.execute(query)

  group_to_result = {}
  for (k, v) in result:
    group_to_result[k] = v
  group_to_result.pop('binned_age')
  return group_to_result

def get_relative_error_per_group(unnoised_result, noised_result):
  unscaled_error = {}
  scaled_error = {}
  for (k, v) in unnoised_result:
    unscaled_error[k] = v - noised_result[k]
    scaled_error[k] = unscaled_error[k] / v
  return unscaled_error, scaled_error

def get_scaled_error_per_group(err, result): 
  group_to_scaled_err = {}
  for (k, v) in result.items():
    group_to_scaled_err[k] = err / v
  return group_to_scaled_err

def get_abs_error(query, epsilon, alpha, meta_path):
  privacy = Privacy(epsilon=epsilon)
  privacy.mechanisms.map[Stat.count] = Mechanism.laplace
  reader = snsql.from_connection(data, privacy=privacy, metadata=meta_path)
  return reader.get_simple_accuracy(query, alpha=alpha)[1]

def alpha_to_percent_conf(alpha):
  return f"{int((1-alpha) * 100)}%"

def compute_all_stats(epsilon, alpha, groupby_keys, data, query, meta_path):
  percent_conf = alpha_to_percent_conf(alpha)
  unnoised_result = get_unnoised_result(data)
  noised_result = get_noised_result(epsilon, data, query, meta_path)
  df = pd.DataFrame(groupby_keys, columns=["Age bin"])
  df.set_index('Age bin')
  key_col = df["Age bin"]
  df["Exact result"] = key_col.apply(lambda age_bin: unnoised_result.get(age_bin, 0))
  df["Noised result"] = key_col.apply(lambda age_bin: noised_result.get(age_bin, 0))
  df[percent_conf + " error"] = get_abs_error(query, epsilon, alpha, meta_path)
  df[percent_conf + " min"] = (df["Noised result"] - df[percent_conf + " error"]).astype('int')
  df[percent_conf + " max"] = (df["Noised result"] + df[percent_conf + " error"]).astype('int')
  df["True error"] = abs(df["Noised result"] - df["Exact result"])
  df["True error %"] = abs(100.0 * (df["Noised result"] - df["Exact result"]) / df["Exact result"])
  df[percent_conf + " error %"] = 100.0 * df[percent_conf + " error"] / df["Noised result"]

  # Do math: expectation of abs(laplace(1/epsi))
  df[f"Expected error"] = 1.0 / epsilon
  df[f"Expected error %"] = 100.0 * df[f"Expected error"] / df["Noised result"]
  return df

########################
# Viz helper functions

def gen_tradeoff_chart(epsilon_choice, line_plot_df):
  # Tutorial from https://altair-viz.github.io/gallery/multiline_tooltip.html
  source = line_plot_df
  # The basic line
  line_plot = alt.Chart(source) \
    .mark_line() \
    .encode(x='epsilon', y='percent_error')

  # Create a selection that chooses the nearest point & selects based on x-value
  nearest = alt.selection(type='single', nearest=True, on='mouseover',
                          fields=['epsilon'], empty='none')

  # Transparent selectors across the chart. This is what tells us
  # the x-value of the cursor
  selectors = alt.Chart(source).mark_point().encode(
      x='epsilon:Q',
      opacity=alt.value(0),
  ).add_selection(
      nearest
  )

  # Draw points on the line, and highlight based on selection
  points = line_plot.mark_point().encode(
      opacity=alt.condition(nearest, alt.value(1), alt.value(0))
  )

  # Draw text labels near the points, and highlight based on selection
  text = line_plot.mark_text(align='left', dx=5, dy=-5).encode(
      text=alt.condition(nearest, 'percent_error:Q', alt.value(' '))
  )

  # Draw a rule at the location of the selection
  rules = alt.Chart(source).mark_rule(color='gray').encode(
      x='epsilon:Q',
  ).transform_filter(
      nearest
  )

  # Draw a rule over the current epsilon selection
  # Guide: https://github.com/altair-viz/altair/issues/1124#issuecomment-417714535
  chosen_epsilon_rule = alt.Chart(source).mark_rule(color='red').encode(
      x='epsilon_choice:Q'
  ).transform_calculate(
    epsilon_choice=f"{epsilon_choice}"
  )

  st.altair_chart(line_plot + selectors + points + rules + text + chosen_epsilon_rule, use_container_width=True)

########################
# Load and process data

# Precompute data. TODO: Use st.cache() and figure out the warnings.
if 'data' not in st.session_state:
  st.session_state.data = load_data(CSV_PATH)
  data = st.session_state.data

  # Used as the denominator for line graph.
  noised_result_reference = get_noised_result(epsilon=1.0, data=data, query=QUERY, meta_path=META_PATH)
  st.session_state.min_denom_reference = min(noised_result_reference.values())

  # Graph data. X = epsilon. Y = Max scaled alpha error percent w.r.t. reference 1.0 epsilon data.
  x = np.linspace(MIN_EPS_VAL, MAX_EPS_VAL, num = int((MAX_EPS_VAL - MIN_EPS_VAL) / EPS_STEP_SIZE))
  y = np.array([get_abs_error(QUERY, eps, ALPHA, META_PATH) for eps in x]) \
    / st.session_state.min_denom_reference \
    * 100.0
  st.session_state.line_plot_df = pd.DataFrame({"epsilon": x, "percent_error": y})

data = st.session_state.data
unnoised_result = get_unnoised_result(data)
percent_conf = alpha_to_percent_conf(ALPHA)
# Assume group by keys are known beforehand.
groupby_keys = sorted(list(unnoised_result.keys()))



#######################
# Actual UI below
epsilon = st.sidebar.slider('epsilon', min_value=MIN_EPS_VAL, max_value=MAX_EPS_VAL, step=EPS_STEP_SIZE)
all_stats = compute_all_stats(epsilon, ALPHA, groupby_keys, data, QUERY, META_PATH)

st.title("Differential Privacy: Visualizing privacy-utility tradeoff")
st.markdown("**Query.** We count the number of people in 5 pre-defined age bins. The dataset is [PUMS1000](https://github.com/opendp/dp-test-datasets/tree/master/data/PUMS_california_demographics_1000).")
st.markdown("**UI overview.** We have two views. The [data user view](#data-user-view) is what you publish to the users requesting the noised statistics. "
  "The [analyst view](#analyst-view) is used by the data curator to figure out the right privacy-utility tradeoff.")
st.markdown("**Recommended experiment.** In this example, epsilon = **1.0** achieves percent error < 10%. Try the following steps:\n"
  "- Adjust the slider on the left nav bar until epsilon = 1.0. \n"
  "- Check that the percent error in the [privacy-utility tradeoff graph](#privacy-utility-tradeoff) is < 10%. \n"
  "- Check in the [data user view](#data-user-view) that the estimated range of the true result is narrow and comparable to the noised result.")

st.markdown("## Data user view")
st.markdown("**Section description**. This section is what you will show to the data users to (1) show the noised result and (2) describe the accuracy of the result.")
st.markdown(f"**Accuracy**. The noised group by results is shown under \"Noised result\". The {percent_conf} confidence estimate of the true data is also given.")
st.markdown(f"**Privacy**. Epsilon = **{epsilon}**. The lower this is, the better-protected users' privacy is. "
  f"For reference, US census used [19.61](https://desfontain.es/privacy/real-world-differential-privacy.html)")

all_stats[percent_conf + " min-max true result"] = all_stats[percent_conf + " min"].astype(str) + "-" + all_stats[percent_conf + " max"].astype(str)
st.table(all_stats[[
  "Age bin",
  "Noised result",
  percent_conf + " min-max true result"]])

st.markdown("## Analyst view")
st.markdown("This internal view should NOT be shown to the data users. This is only for the analysts.")
st.markdown("### Privacy-utility tradeoff")
st.markdown("**Usage guide.** Increase the epsilon so the percent_error falls under the desired threshold (e.g. 10%). ")
st.markdown(f"**Y-axis.** **Numerator =** {percent_conf} error (based on laplace distribution math, not based on true data), and "
  f"**denominator =** the min group value that has been noised by epsilon = 1.0 once."
  f" For this session, the denominator is {st.session_state.min_denom_reference}.")
st.markdown(f"**Why a complicated denominator?** We do NOT use the true value as the denominator. "
  f"If the analyst publishes both epsilon and the percent error based on the true result, an attacker can exactly know the true result. "
  "Consuming an untracked budget of 1.0 once is a compromise to get the easier-to-understand scaled error.")
gen_tradeoff_chart(epsilon, st.session_state.line_plot_df)

st.markdown("### Error")
st.markdown("**Usage guide.** Here you can see the results before and after noise has been added. Expected error is the expectation of absolute laplace R.V.")
st.table(all_stats[[
  "Age bin",
  "Exact result",
  "Noised result",
  "True error",
  "Expected error",
  f"{percent_conf} error"]])

st.markdown("### Percent error")
st.markdown("**Usage guide.** Just like the previous view, but normalized. "
  "True error uses true data as the denominator. The rest uses noised data as the denominator.")
st.table(all_stats[[
  "Age bin",
  "Exact result",
  "Noised result",
  "True error %",
  "Expected error %",
  percent_conf + " error %"]])
