response = llm.invoke(prompt)
    return extract_float(response.content)

    response = await llm.ainvoke(prompt)
    return extract_float(response.content)